import torch
from torch import nn
import torch.nn.functional as F
from voxel_uncertainty import voxel_uncertainty, weighted_expectation_of_entropy, expectation_of_entropy, \
    entropy_of_expected
import numpy as np
import json
import math


class EnsembleUnet:
    def __init__(self, Unets, path_stats=None):
        self.Unets = Unets  # list of Unets

        if path_stats is not None:
            with open(path_stats, 'r') as f:
                stat_dict = json.load(f)
            self.stat_dict = stat_dict

    def get_stats(self, index):
        stat_i = {"mu_0": self.stat_dict[f"model{index}_mu_0"], "std_0": self.stat_dict[f"model{index}_std_0"],
                  "mu_1": self.stat_dict[f"model{index}_mu_1"], "std_1": self.stat_dict[f"model{index}_std_1"]}
        return stat_i

    def get_all_stats(self):
        return self.stat_dict

    def get_model(self, index):
        return self.Unets[index]

    def combine_outputs(self, outputs, combination, th):
        if combination == 'mean':
            return torch.stack(outputs).mean(axis=0)
        elif combination == 'majority_vote':
            # print(outputs[0].shape)
            stacked_outputs = torch.stack(outputs)
            stacked_outputs[stacked_outputs >= th] = 1
            stacked_outputs[stacked_outputs < th] = 0
            # print(stacked_outputs.shape)

            # mode, indices = torch.mode(stacked_outputs.long(), dim=0)
            # n_models = stacked_outputs.shape[0]
            mode = torch.sum(stacked_outputs, dim=0)

            mode[mode <= 1] = 0  # only 1 or 0 models has predicted 1 -> 0
            mode[mode >= 2] = 1  # at least 2 models or 3 have predicted 1 -> 1

            # one_hot_output = nn.functional.one_hot(mode.long())
            # print(one_hot_output.shape)
            # majority[torch.arange(majority.shape[0]), mode.argmax(dim=1)] = 1
            # Return the majority one-hot vector as the output
            return mode.numpy()

    def majority_vote(self, probs, th):
        #print('-' * 20)
        #print(probs.shape)
        mask = ~torch.eq(probs, -1)  # create boolean mask for values different of -1
        #print(mask.shape)
        n_models = probs.shape[0]
        new_probs = torch.zeros_like(probs)

        new_probs[probs >= th] = 1
        new_probs[probs < th] = 0

        non_negative_count = torch.sum(mask, axis=0)  # count non-negative values
        #print(non_negative_count.shape)
        mode = torch.sum(new_probs, dim=0)

        mode[mode > non_negative_count // 2] = 1
        mode[mode < non_negative_count // 2] = 0

        remaining = torch.eq(mode, non_negative_count // 2)
        remaining_count = torch.sum(remaining)

        if remaining_count > 1:
            #print('remaining', remaining_count)
            confidences = torch.abs(probs[:, remaining] - th)
            argmax = torch.argmax(confidences, dim=0)
            #print(argmax)
            mode[remaining] = new_probs[argmax, remaining]
        return mode

    def mean_vote(self, probs, th):

        mask = ~torch.eq(probs, -1)  # create boolean mask for values different of -1
        n_models = probs.shape[0]
        new_probs = torch.zeros_like(probs)
        new_probs[probs >= th] = 1
        new_probs[probs < th] = 0
        non_negative_count = torch.sum(mask, axis=0)  # count non-negative values
        mode = torch.sum(new_probs, dim=0) / non_negative_count

        mode[mode >= th] = 1
        mode[mode < th] = 0

        return mode

    def weighted_mean_vote(self, probs, mask, weight_unconfident, th):
        weighted_mask = mask.float() + (1 - mask.float()) * weight_unconfident
        n_models = probs.shape[0]
        new_probs = torch.zeros_like(probs)
        new_probs[probs >= th] = 1
        new_probs[probs < th] = 0
        non_negative_count = torch.sum(weighted_mask, axis=0)
        mode = torch.sum(new_probs * weighted_mask, dim=0) / non_negative_count

        mode[mode >= th] = 1
        mode[mode < th] = 0

        return mode

    def logprob_mean_vote(self, probs, confs, th, temperature):

        # weights = confs / torch.sum(confs, dim=0, keepdim=True)
        # print(confs[:,0,0,0])
        weights = F.softmax(confs / temperature, dim=0)
        #print(weights[:, 50, 50, 50])
        # Compute the weighted mean predictions
        mean_probs = torch.sum(weights * probs, dim=0)

        # Apply the threshold

        return mean_probs

    def weighted_std(self, probs, weighted_mean, confs, temperature):
        weights = F.softmax(confs / temperature, dim=0)
        # weights = confs / torch.sum(confs, dim=0, keepdim=True)
        squared_sum = torch.sum(weights ** 2, dim=0)
        var = torch.sum(weights * (probs - weighted_mean) ** 2, dim=0) * 1 / (1 - squared_sum)
        std = torch.sqrt(var)
        # print(std)
        return std

    '''
        def majority_vote(self, probs, th):
            n_models = probs.shape[0]
            new_probs = torch.zeros_like(probs)
            mode = torch.zeros_like(probs[0, :])

            new_probs[probs >= th] = 1
            new_probs[probs < th] = 0
            mode = torch.sum(new_probs, dim=0)

            mode[mode > n_models // 2] = 1
            mode[mode < n_models // 2] = 0

            remaining = mode == n_models // 2
            print(mode.shape)
            print(torch.sum(remaining))
            print(n_models)
            if torch.sum(remaining) > 1:
                confidences = torch.abs(probs[:, remaining] - th)
                print(confidences.shape)
                argmax = torch.argmax(confidences, dim=0)
                print(argmax.shape)

                mode[remaining] = new_probs[argmax]
            return mode
    '''

    def combine_confidences(self, confidences, outputs, th, combination, alpha=2, unc_measure='EOE',
                            weight_unconfident=0.5, temperature=1.0):

        confidences_filtered = []
        final_confidences = torch.zeros_like(confidences[0])
        final_outputs = torch.zeros_like(confidences[0])

        values_for_shift = [1e0, 1e3, 1e6, 1e9]
        if combination == 'majority_vote_separated':
            for i in range(len(self.Unets)):
                stats = self.get_stats(i)
                mask_0 = outputs[i] < th  # check which class the model predicted
                mask_1 = outputs[i] >= th
                conf = confidences[i]
                # check if in or out of distribution for this confidence level and for this particular class
                conf[mask_0] = (conf[mask_0] > (stats['mu_0'] - alpha * stats['std_0'])).float()
                conf[mask_1] = (conf[mask_1] > (stats['mu_1'] - alpha * stats['std_1'])).float()
                confidences_filtered.append(conf)

            stacked_confidences_filtered = torch.stack(confidences_filtered)  # stack the 3 models
            count = torch.sum(stacked_confidences_filtered,
                              dim=0).squeeze()  # count the number of confident model for each prediction
            shape = count.shape
            count = count.view(-1)
            values, indices = torch.sort(count,
                                         descending=True)  # sort them to get first the predictions with 3 confident models, then 2 and so on

            for i in reversed(range(len(self.Unets) + 1)):
                mask = values == i  # indices with i confident models
                indices_mask = indices[mask]  # get the corresponding indices
                probs = torch.stack(outputs)

                confidences_flatten = stacked_confidences_filtered.view(stacked_confidences_filtered.shape[0], -1)[:,
                                      indices_mask]  # get the confidences of the corresponding predictions
                probs = probs.view(probs.shape[0], -1)[:, indices_mask]  # get the corresponding predictions
                if i != 0:  # if i!=0 need to only select the confident models
                    model_mask = confidences_flatten == 1  # look which models are confident
                    # print(model_mask.size())
                    probs = probs * model_mask  # set to 0 the predictions for unconfident models
                    probs = torch.where(probs == 0, -1,
                                        probs)  # set to -1 the predictions for unconfident models (see after for compting uncertainty)
                else:
                    probs = probs

                if len(indices_mask) > 0:  # if at least one prediction
                    # probs = stacked_outputs.view(stacked_outputs.shape[0], -1)[:, indices_mask]
                    #print(i, probs.shape)
                    unc = -voxel_uncertainty(probs=probs, measure=unc_measure) + values_for_shift[i]
                    final_confidences.view(-1)[indices_mask] = unc  # fill the uncertainty at the corresponding indices
                    preds_final = self.majority_vote(probs, th)
                    final_outputs.view(-1)[indices_mask] = preds_final

            return final_confidences.numpy(), final_outputs.numpy()

        elif combination == 'majority_vote_combined':
            for i in range(len(self.Unets)):
                stats = self.get_stats(i)
                mask_0 = outputs[i] < th  # check which class the model predicted
                mask_1 = outputs[i] >= th
                conf = confidences[i]
                # check if in or out of distribution for this confidence level and for this particular class
                conf[mask_0] = (conf[mask_0] > (stats['mu_0'] - alpha * stats['std_0'])).float()
                conf[mask_1] = (conf[mask_1] > (stats['mu_1'] - alpha * stats['std_1'])).float()
                confidences_filtered.append(conf)

            stacked_confidences_filtered = torch.stack(confidences_filtered)  # stack the 3 models
            count = torch.sum(stacked_confidences_filtered,
                              dim=0).squeeze()  # count the number of confident model for each prediction
            shape = count.shape
            count = count.view(-1)
            values, indices = torch.sort(count,
                                         descending=True)  # sort them to get first the predictions with 3 confident models, then 2 and so on

            for i in reversed(range(len(self.Unets) + 1)):
                mask = values == i  # indices with i confident models
                indices_mask = indices[mask]  # get the corresponding indices
                probs = torch.stack(outputs)

                confidences_flatten = stacked_confidences_filtered.view(stacked_confidences_filtered.shape[0], -1)[:,
                                      indices_mask]  # get the confidences of the corresponding predictions
                probs = probs.view(probs.shape[0], -1)[:, indices_mask]  # get the corresponding predictions
                if i != 0:  # if i!=0 need to only select the confident models
                    model_mask = confidences_flatten == 1  # look which models are confident
                    # print(model_mask.size())
                    probs = probs * model_mask  # set to 0 the predictions for unconfident models
                    probs = torch.where(probs == 0, -1,
                                        probs)  # set to -1 the predictions for unconfident models (see after for compting uncertainty)
                else:
                    probs = probs

                if len(indices_mask) > 0:  # if at least one prediction
                    # probs = stacked_outputs.view(stacked_outputs.shape[0], -1)[:, indices_mask]
                    #print(i, probs.shape)
                    unc = -voxel_uncertainty(probs=probs, measure=unc_measure)
                    #print(unc)
                    final_confidences.view(-1)[indices_mask] = unc  # fill the uncertainty at the corresponding indices
                    preds_final = self.majority_vote(probs, th)
                    final_outputs.view(-1)[indices_mask] = preds_final

            return final_confidences.numpy(), final_outputs.numpy()

        elif combination == 'mean_vote_combined':
            for i in range(len(self.Unets)):
                stats = self.get_stats(i)
                mask_0 = outputs[i] < th  # check which class the model predicted
                mask_1 = outputs[i] >= th
                conf = confidences[i]
                # check if in or out of distribution for this confidence level and for this particular class
                conf[mask_0] = (conf[mask_0] > (stats['mu_0'] - alpha * stats['std_0'])).float()
                conf[mask_1] = (conf[mask_1] > (stats['mu_1'] - alpha * stats['std_1'])).float()
                confidences_filtered.append(conf)

            stacked_confidences_filtered = torch.stack(confidences_filtered)  # stack the 3 models
            count = torch.sum(stacked_confidences_filtered,
                              dim=0).squeeze()  # count the number of confident model for each prediction
            shape = count.shape
            count = count.view(-1)
            values, indices = torch.sort(count,
                                         descending=True)  # sort them to get first the predictions with 3 confident models, then 2 and so on

            for i in reversed(range(len(self.Unets) + 1)):
                mask = values == i  # indices with i confident models
                indices_mask = indices[mask]  # get the corresponding indices
                probs = torch.stack(outputs)

                confidences_flatten = stacked_confidences_filtered.view(stacked_confidences_filtered.shape[0], -1)[:,
                                      indices_mask]  # get the confidences of the corresponding predictions
                probs = probs.view(probs.shape[0], -1)[:, indices_mask]  # get the corresponding predictions
                if i != 0:  # if i!=0 need to only select the confident models
                    model_mask = confidences_flatten == 1  # look which models are confident
                    # print(model_mask.size())
                    probs = probs * model_mask  # set to 0 the predictions for unconfident models
                    probs = torch.where(probs == 0, -1,
                                        probs)  # set to -1 the predictions for unconfident models (see after for compting uncertainty)
                else:
                    probs = probs

                if len(indices_mask) > 0:  # if at least one prediction
                    # probs = stacked_outputs.view(stacked_outputs.shape[0], -1)[:, indices_mask]
                    #print(i, probs.shape)
                    unc = -voxel_uncertainty(probs=probs, measure=unc_measure)
                    #print(unc)
                    final_confidences.view(-1)[indices_mask] = unc  # fill the uncertainty at the corresponding indices
                    preds_final = self.mean_vote(probs, th)
                    final_outputs.view(-1)[indices_mask] = preds_final

            return final_confidences.numpy(), final_outputs.numpy()

        elif combination == 'weighted_mean_vote_combined':
            for i in range(len(self.Unets)):
                stats = self.get_stats(i)
                mask_0 = outputs[i] < th  # check which class the model predicted
                mask_1 = outputs[i] >= th
                conf = confidences[i]
                # check if in or out of distribution for this confidence level and for this particular class
                conf[mask_0] = (conf[mask_0] > (stats['mu_0'] - alpha * stats['std_0'])).float()
                conf[mask_1] = (conf[mask_1] > (stats['mu_1'] - alpha * stats['std_1'])).float()
                confidences_filtered.append(conf)

            stacked_confidences_filtered = torch.stack(confidences_filtered)  # stack the 3 models
            count = torch.sum(stacked_confidences_filtered,
                              dim=0).squeeze()  # count the number of confident model for each prediction
            shape = count.shape
            count = count.view(-1)
            values, indices = torch.sort(count,
                                         descending=True)  # sort them to get first the predictions with 3 confident models, then 2 and so on

            for i in reversed(range(len(self.Unets) + 1)):
                mask = values == i  # indices with i confident models
                indices_mask = indices[mask]  # get the corresponding indices
                probs = torch.stack(outputs)

                confidences_flatten = stacked_confidences_filtered.view(stacked_confidences_filtered.shape[0], -1)[:,
                                      indices_mask]  # get the confidences of the corresponding predictions
                probs = probs.view(probs.shape[0], -1)[:, indices_mask]  # get the corresponding predictions
                model_mask = confidences_flatten == 1  # look which models are confident

                # probs = probs * model_mask  # set to 0 the predictions for unconfident models
                # probs = torch.where(probs == 0, -1, probs)  # set to -1 the predictions for unconfident models (see after for compting uncertainty)
                probs = probs

                if len(indices_mask) > 0:  # if at least one prediction
                    # probs = stacked_outputs.view(stacked_outputs.shape[0], -1)[:, indices_mask]
                    unc = -voxel_uncertainty(probs=probs, mask=model_mask, measure=unc_measure,
                                             weight_unconfident=weight_unconfident)
                    final_confidences.view(-1)[indices_mask] = unc  # fill the uncertainty at the corresponding indices
                    preds_final = self.weighted_mean_vote(probs, model_mask, weight_unconfident, th)
                    final_outputs.view(-1)[indices_mask] = preds_final

            return final_confidences.numpy(), final_outputs.numpy()

        elif combination == 'logprob_mean_vote':

            # outputs = [M, H, W, Z], confidences = [M, H, W, Z]
            confidences = torch.stack(confidences)
            confidences = confidences - torch.min(
                confidences) + 1e-2  # only positive confidences (slight margin to avoid 0 in weighted mean)

            probs = torch.stack(outputs)

            predictions = self.logprob_mean_vote(probs, confidences, th, temperature)
            predictions_th = torch.where(predictions >= th, torch.tensor(1.0), torch.tensor(0.0))
            # unc = -voxel_uncertainty(probs=probs, measure=unc_measure)#.numpy()

            unc = torch.abs(predictions - th)
            # unc = torch.amax(confidences, dim=0)

            # unc = -weighted_expectation_of_entropy(probs, confidences, epsilon=1e-10)
            return unc.numpy(), predictions_th.numpy()

        elif combination == 'logprob_mean_vote_improved':

            # outputs = [M, H, W, Z], confidences = [M, H, W, Z]
            confidences = torch.stack(confidences)
            # no_confidences = torch.ones(confidences.shape) #
            # confidences = confidences - torch.min(confidences) + 1e-8  # only positive confidences (slight margin to avoid 0 in weighted mean)

            probs = torch.stack(outputs)

            # predictions = self.logprob_mean_vote(probs, no_confidences, th)
            predictions = self.logprob_mean_vote(probs, confidences, th, temperature)
            # print(predictions-log_predictions)
            predictions_th = torch.where(predictions >= th, torch.tensor(1.0), torch.tensor(0.0))
            # unc = -voxel_uncertainty(probs=probs, measure=unc_measure)#.numpy()

            # weighted_std = self.weighted_std(probs, predictions, confidences)
            # conf = torch.abs(predictions-th)/(1+weighted_std)

            # t = th*(predictions_th == 1) + (1-th)*(predictions_th==0)
            conf = torch.abs(predictions - th)  # /(1+self.weighted_std)

            # weighted_std = torch.where(conf<=0.1, weighted_std, torch.tensor(0))
            # conf = conf/(1+weighted_std)
            # conf = - torch.abs(predictions_th - predictions)#/t
            # conf = conf - alpha * weighted_std * (predictions_th==1)
            # unc = torch.amax(confidences, dim=0)

            # unc = -weighted_expectation_of_entropy(probs, confidences, epsilon=1e-10)
            return conf.numpy(), predictions_th.numpy()

        elif combination == 'logprob_mean_vote_std':

            confidences = torch.stack(confidences)
            probs = torch.stack(outputs)
            predictions = self.logprob_mean_vote(probs, confidences, th, temperature)
            predictions_th = torch.where(predictions >= th, torch.tensor(1.0), torch.tensor(0.0))
            weighted_std = self.weighted_std(probs, predictions, confidences, temperature)
            conf = torch.abs(predictions - th) / (1 + alpha * weighted_std)

            return conf.numpy(), predictions_th.numpy()

        elif combination == 'logprob_mean_vote_std_density':

            confidences = torch.stack(confidences)
            probs = torch.stack(outputs)
            predictions = self.logprob_mean_vote(probs, confidences, th, temperature)
            predictions_th = torch.where(predictions >= th, torch.tensor(1.0), torch.tensor(0.0))
            weighted_std = self.weighted_std(probs, predictions, confidences, temperature)
            conf = torch.abs(predictions - th) / (1 + alpha * weighted_std)

            conf = conf / torch.max(conf)  # conf between 0 and 1
            conf_shape = conf.shape
            conf = conf.flatten()

            mean_confidences = torch.mean(confidences, dim=0)
            shape = mean_confidences.shape
            mean_confidences = mean_confidences.flatten()
            indices = torch.arange(mean_confidences.shape)
            indices_0 = indices[predictions_th.flatten() == 0]
            indices_1 = indices[predictions_th.flatten() == 1]

            _, indices_conf = torch.sort(conf, descending=True)

            # First indice 0
            max_mean_0 = torch.max(mean_confidences[indices_0])
            min_mean_0 = torch.min(mean_confidences[indices_0])
            mean_confidences_0 = mean_confidences[indices_0] - min_mean_0 + 1e-8
            conf_0 = conf[indices_0]
            indices_conf_0 = indices_conf[indices_0]
            new_conf_0 = conf_0 * mean_confidences_0
            _, new_indices_conf_0 = torch.sort(new_values_conf_0, descending=True)

            # Second indice 1
            max_mean_1 = torch.max(mean_confidences[indices_1])
            min_mean_1 = torch.min(mean_confidences[indices_1])
            mean_confidences_1 = mean_confidences[indices_1] - min_mean_1 + 1e-8

            return conf.numpy(), predictions_th.numpy()

        elif combination == 'logprob_mean_vote_th':

            # outputs = [M, H, W, Z], confidences = [M, H, W, Z]
            confidences = torch.stack(confidences)
            # no_confidences = torch.ones(confidences.shape) #
            # confidences = confidences - torch.min(confidences) + 1e-8  # only positive confidences (slight margin to avoid 0 in weighted mean)

            probs = torch.stack(outputs)

            # predictions = self.logprob_mean_vote(probs, no_confidences, th)
            predictions = self.logprob_mean_vote(probs, confidences, th, temperature)
            # print(predictions-log_predictions)
            predictions_th = torch.where(predictions >= th, torch.tensor(1.0), torch.tensor(0.0))
            # unc = -voxel_uncertainty(probs=probs, measure=unc_measure)#.numpy()

            weighted_std = self.weighted_std(probs, predictions, confidences, temperature)
            # conf = torch.abs(predictions-th)/(1+weighted_std)

            t = th * (predictions_th == 1) + (1 - th) * (predictions_th == 0)
            conf = torch.abs(predictions - th) / t

            # weighted_std = torch.where(conf<=0.1, weighted_std, torch.tensor(0))
            # conf = conf/(1+weighted_std)
            # conf = - torch.abs(predictions_th - predictions)#/t
            # conf = conf - alpha * weighted_std * (predictions_th==1)
            # unc = torch.amax(confidences, dim=0)

            # unc = -weighted_expectation_of_entropy(probs, confidences, epsilon=1e-10)
            return conf.numpy(), predictions_th.numpy()

        elif combination == 'simple_mean_vote':

            # outputs = [M, H, W, Z], confidences = [M, H, W, Z]
            confidences = torch.stack(confidences)
            # no_confidences = torch.ones(confidences.shape) #
            # confidences = confidences - torch.min(confidences) + 1e-8  # only positive confidences (slight margin to avoid 0 in weighted mean)

            probs = torch.stack(outputs)

            # predictions = self.logprob_mean_vote(probs, no_confidences, th)
            predictions = self.logprob_mean_vote(probs, confidences, th)
            # print(predictions-log_predictions)
            predictions_th = torch.where(predictions >= th, torch.tensor(1.0), torch.tensor(0.0))
            # unc = -voxel_uncertainty(probs=probs, measure=unc_measure)#.numpy()

            # weighted_std = self.weighted_std(probs, predictions, confidences)
            # conf = torch.abs(predictions-th)/(1+weighted_std)

            # t = th*(predictions_th == 1) + (1-th)*(predictions_th==0)
            conf = torch.abs(predictions - th)  # /(1+self.weighted_std)

            # weighted_std = torch.where(conf<=0.1, weighted_std, torch.tensor(0))
            # conf = conf/(1+weighted_std)
            # conf = - torch.abs(predictions_th - predictions)#/t
            # conf = conf - alpha * weighted_std * (predictions_th==1)
            # unc = torch.amax(confidences, dim=0)

            # unc = -weighted_expectation_of_entropy(probs, confidences, epsilon=1e-10)
            return conf.numpy(), predictions_th.numpy()

        elif combination == 'majority_vote_combined_confidents':
            for i in range(len(self.Unets)):
                stats = self.get_stats(i)
                mask_0 = outputs[i] < th  # check which class the model predicted
                mask_1 = outputs[i] >= th
                conf = confidences[i]
                # check if in or out of distribution for this confidence level and for this particular class
                conf[mask_0] = (conf[mask_0] > (stats['mu_0'] - alpha * stats['std_0'])).float()
                conf[mask_1] = (conf[mask_1] > (stats['mu_1'] - alpha * stats['std_1'])).float()
                confidences_filtered.append(conf)

            stacked_confidences_filtered = torch.stack(confidences_filtered)  # stack the 3 models
            count = torch.sum(stacked_confidences_filtered,
                              dim=0).squeeze()  # count the number of confident model for each prediction
            shape = count.shape
            count = count.view(-1)
            values, indices = torch.sort(count,
                                         descending=True)  # sort them to get first the predictions with 3 confident models, then 2 and so on

            offset = 1e6
            for i in reversed(range(len(self.Unets) + 1)):
                mask = values == i  # indices with i confident models
                indices_mask = indices[mask]  # get the corresponding indices
                probs = torch.stack(outputs)

                confidences_flatten = stacked_confidences_filtered.view(stacked_confidences_filtered.shape[0], -1)[:,
                                      indices_mask]  # get the confidences of the corresponding predictions
                probs = probs.view(probs.shape[0], -1)[:, indices_mask]  # get the corresponding predictions
                if i != 0:  # if i!=0 need to only select the confident models
                    model_mask = confidences_flatten == 1  # look which models are confident
                    # print(model_mask.size())
                    probs = probs * model_mask  # set to 0 the predictions for unconfident models
                    probs = torch.where(probs == 0, -1,
                                        probs)  # set to -1 the predictions for unconfident models (see after for compting uncertainty)
                else:
                    probs = probs
                    offset = 0

                if len(indices_mask) > 0:  # if at least one prediction
                    # probs = stacked_outputs.view(stacked_outputs.shape[0], -1)[:, indices_mask]
                    #print(i, probs.shape)
                    unc = -voxel_uncertainty(probs=probs, measure=unc_measure) + offset

                    final_confidences.view(-1)[indices_mask] = unc  # fill the uncertainty at the corresponding indices
                    preds_final = self.majority_vote(probs, th)
                    final_outputs.view(-1)[indices_mask] = preds_final

            return final_confidences.numpy(), final_outputs.numpy()

        elif combination == 'mean_vote_combined_confidents':
            for i in range(len(self.Unets)):
                stats = self.get_stats(i)
                mask_0 = outputs[i] < th  # check which class the model predicted
                mask_1 = outputs[i] >= th
                conf = confidences[i]
                # check if in or out of distribution for this confidence level and for this particular class
                conf[mask_0] = (conf[mask_0] > (stats['mu_0'] - alpha * stats['std_0'])).float()
                conf[mask_1] = (conf[mask_1] > (stats['mu_1'] - alpha * stats['std_1'])).float()
                confidences_filtered.append(conf)

            stacked_confidences_filtered = torch.stack(confidences_filtered)  # stack the 3 models
            count = torch.sum(stacked_confidences_filtered,
                              dim=0).squeeze()  # count the number of confident model for each prediction
            shape = count.shape
            count = count.view(-1)
            values, indices = torch.sort(count,
                                         descending=True)  # sort them to get first the predictions with 3 confident models, then 2 and so on

            offset = 1e6
            for i in reversed(range(len(self.Unets) + 1)):
                mask = values == i  # indices with i confident models
                indices_mask = indices[mask]  # get the corresponding indices
                probs = torch.stack(outputs)

                confidences_flatten = stacked_confidences_filtered.view(stacked_confidences_filtered.shape[0], -1)[:,
                                      indices_mask]  # get the confidences of the corresponding predictions
                probs = probs.view(probs.shape[0], -1)[:, indices_mask]  # get the corresponding predictions
                if i != 0:  # if i!=0 need to only select the confident models
                    model_mask = confidences_flatten == 1  # look which models are confident
                    # print(model_mask.size())
                    probs = probs * model_mask  # set to 0 the predictions for unconfident models
                    probs = torch.where(probs == 0, -1,
                                        probs)  # set to -1 the predictions for unconfident models (see after for compting uncertainty)
                else:
                    probs = probs
                    offset = 0

                if len(indices_mask) > 0:  # if at least one prediction
                    # probs = stacked_outputs.view(stacked_outputs.shape[0], -1)[:, indices_mask]
                    #print(i, probs.shape)
                    unc = -voxel_uncertainty(probs=probs, measure=unc_measure) + offset

                    final_confidences.view(-1)[indices_mask] = unc  # fill the uncertainty at the corresponding indices
                    preds_final = self.mean_vote(probs, th)
                    final_outputs.view(-1)[indices_mask] = preds_final

            return final_confidences.numpy(), final_outputs.numpy()


        elif combination == 'majority_vote_half_separated':  # 3 et 2 sont ensemble puis on met 1 puis 0
            for i in range(len(self.Unets)):
                stats = self.get_stats(i)

                mask_0 = outputs[i] < th
                mask_1 = outputs[i] >= th

                conf = confidences[i]
                conf[mask_0] = (conf[mask_0] > (stats['mu_0'] - alpha * stats['std_0'])).float()
                conf[mask_1] = (conf[mask_1] > (stats['mu_1'] - alpha * stats['std_1'])).float()
                confidences_filtered.append(conf)  # 1 if in-distribution, 0 otherwise
            stacked_confidences_filtered = torch.stack(confidences_filtered)
            count = torch.sum(stacked_confidences_filtered, dim=0).squeeze()
            #print('count', count.shape)
            shape = count.shape
            count = count.flatten()
            values, indices = torch.sort(count, descending=True)

            for i in reversed(range(math.ceil(len(self.Unets) / 2))):
                if i >= len(self.Unets) / 2:
                    mask = values >= i
                else:
                    mask = values == i

                indices_mask = indices[mask]
                confidences_flatten = stacked_confidences_filtered.flatten(1, -1)[maks]
                probs = stacked_outputs.flatten(1, -1)[:, indices_mask]
                if i != 0:
                    model_mask = confidences_flatten == 1
                    probs = probs[model_mask]
                else:
                    probs = probs

                unc = voxel_uncertainty(probs=probs, measure=unc_measure) + values_for_shift[i]
                #print(unc.shape)
                final_confidences[indices_mask] = unc
            # get the indices of the 3 values, order them, 2 values, order them and so on.

            final_confidences = final_confidences.view(shape), outputs

        elif combination == 'EOE':
            return voxel_uncertainty(probs=torch.stack(outputs), measure='EOE')

    def combine_confidencesv2(self, confidences, outputs):
        # outputs shape list of num_models tensors of shape (B,num_classes,H,W,Z)
        # need tensor of shape (num_models, B, H, W, Z, num_classes)
        stacked_outputs = torch.stack(outputs)
        stacked_confidences = torch.tensor(np.array(confidences))

        EoE = voxel_uncertainty(stacked_outputs, measure='EOEv2')

        return EoE.numpy()

    def combine_confidencesv3(self, confidences, outputs):
        # outputs shape list of num_models tensors of shape (B,num_classes,H,W,Z)
        # need tensor of shape (num_models, B, H, W, Z, num_classes)
        stacked_outputs = torch.stack(outputs)
        stacked_confidences = torch.tensor(np.array(confidences))

        EoE = voxel_uncertainty(stacked_confidences, measure='EOEv3')

        return EoE.numpy()
