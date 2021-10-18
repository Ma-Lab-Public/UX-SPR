import argparse
import csv
import os
import sys
import time
from collections import defaultdict

# import boto3
import comet_ml
import pyro
import pyro.distributions as dist
import torch
# from botocore.exceptions import ClientError
from comet_ml import api
from tqdm import tqdm

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(32)
torch.set_num_interop_threads(32)


def calc_perplexity(ids, sample_num=10):
    test_data, test_args = ids.get_test_set()

    log_sum_p = 0
    for tags in test_data['tag']:
        for tag_id in tags:
            log_sum_p += torch.log(torch.Tensor([calc_liklihood(tag_id, sample_num)]))

    perplexity = torch.exp(- log_sum_p / torch.numel(test_data['tag'])).item()

    return perplexity

def download_posterior(exp_key):
    
    file_name = exp_key + '.pkl'
    
    try:
        os.system(f'cp ./pkl_model/{file_name} ./')
    except:
        print(f"Failed to copy {file_name}")

def get_test_data(data_file, test_ids):
    with open(data_file) as f:
        reader = csv.reader(f)
        
        us = []
        ts = []
        ls = []
        tag_matrix = []

        for i, row in enumerate(reader):
            if i in test_ids:
                us.append(int(row[1]))
                ts.append(int(row[2]))
                ls.append(int(row[3]))
                tag_matrix.append([int(t) for t in row[4].split(",")])

    data = {
        'u': torch.LongTensor(us).to(device),
        't': torch.LongTensor(ts).to(device),
        'l': torch.LongTensor(ls).to(device),
        'tag': torch.LongTensor(tag_matrix).to(device)
    }

    return data

def divide_data_by_user(data, posterior):
    user_count = posterior['gamma_q'].shape[1]
    location_count= posterior['beta_q'].shape[1]

    user_location_matrix = torch.zeros(user_count, location_count).to(device)

    for u, l in zip(data['u'], data['l']):
        user_location_matrix[u][l] += 1

    return user_location_matrix

def calc_word_perplexity_given_user(posterior, test_data, sample_size=100):
    n = torch.numel(test_data['tag'])

    alpha_q = posterior['alpha_q'].to(device)
    gamma_q = posterior['gamma_q'].to(device)
    delta_q = posterior['delta_q'].to(device)

    size = torch.LongTensor([sample_size]).to(device)
    likelihood_list = []
    for u, tags in zip(tqdm(test_data['u']), test_data['tag']):
        for tag_id in tags:
            try:
                theta = dist.Dirichlet(alpha_q).sample(size)
                pi = dist.Dirichlet(gamma_q[:, u]).sample(size)
                sigma = dist.Dirichlet(delta_q[:, tag_id]).sample(size)

                # aim: (sample_size, group_count)
                likelihood_list.append((theta * pi * sigma).sum(1).to('cpu'))
            except:
                pass

    log_sum_p = torch.sum(torch.log(torch.stack(likelihood_list).to(device))) / size

    perplexity = torch.exp(- log_sum_p / n)

    return perplexity.to('cpu').item()

def calc_word_perplexity_given_user_with_location_model(posterior, test_data, sample_size=100):
    size = torch.LongTensor([sample_size]).to(device)
    n = torch.numel(test_data['tag'])

    alpha_q = posterior['alpha_q'].to(device)
    beta_q = posterior['beta_q'].to(device)
    gamma_q = posterior['gamma_q'].to(device)
    delta_q = posterior['delta_q'].to(device)
    #zeta_alpha_q = posterior['zeta_alpha_q'].to(device)#2021/09/23
    #zeta_beta_q = posterior['zeta_beta_q'].to(device)#2021/09/23
    zeta_q = posterior['zeta_q'].to(device)#2021/09/23
    epsilon_q = posterior['epsilon_q'].to(device)

    likelihood_list = []
    for u, tags in zip(tqdm(test_data['u']), test_data['tag']):
        for tag_id in tags:
            try:
                theta = dist.Dirichlet(alpha_q).sample(size)
                pi = dist.Dirichlet(gamma_q[:, u]).sample(size)
                phi = dist.Dirichlet(beta_q).sample(size)
                sigma = dist.Dirichlet(delta_q[:, tag_id]).sample(size)
                #eta  = dist.Beta(zeta_beta_q, zeta_beta_q).sample(size)#2021/09/23
                eta  = dist.Dirichlet(zeta_q).sample(size)#2021/09/23
                mu = dist.Dirichlet(epsilon_q[:, tag_id]).sample(size)
                #lmd = dist.Bernoulli(eta).sample()#2021/09/23
                lmd = dist.Multinomial(1, eta).sample()#2021/09/23

                # lmd: (sample_size, location_size)
                # mu: (sample_size, location_size)
                # sigma: (sample_size, group_count)
                # phi: (sample_size, group_count, location_size)
                # pi: (sample_size, group_count)
                # theta: (sample_size, group_count)
                # aim -> (sample_size, group_count, location_size)
                #2021/09/23#likelihood_list.append((((lmd * mu).unsqueeze(1) + (1 - lmd).unsqueeze(1) * sigma.unsqueeze(2)) * phi * pi.unsqueeze(2) * theta.unsqueeze(2)).view(size, -1).sum(1).to('cpu'))
                p_w_given_tlg = (lmd.index_select(2, torch.LongTensor([0]).to(device)) * mu.unsqueeze(2)).unsqueeze(1)+(lmd.index_select(2, torch.LongTensor([1]).to(device))* sigma.unsqueeze(1)).unsqueeze(1)#2021/09/23

                likelihood_list.append((p_w_given_tlg * phi.unsqueeze(3) * (pi * theta).view(size, -1, 1, 1)).view(size, -1).sum(1).to('cpu'))#2021/09/23
            except:
                pass
    
            
    log_sum_p = torch.sum(torch.log(torch.stack(likelihood_list).to(device))) / size

    perplexity = torch.exp(- log_sum_p / n)

    return perplexity.to('cpu').item()

def calc_word_perplexity_given_user_with_timeaware_model(posterior, test_data, sample_size=100):
    size = torch.LongTensor([sample_size]).to(device)
    n = torch.numel(test_data['tag'])

    alpha_q = posterior['alpha_q'].to(device)
    gamma_q = posterior['gamma_q'].to(device)
    kappa_q = posterior['kappa_q'].to(device)
    delta_q = posterior['delta_q'].to(device)
    zeta_q = posterior['zeta_q'].to(device)
    iota_q = posterior['iota_q'].to(device)

    likelihood_list = []
    for u, tags in zip(tqdm(test_data['u']), test_data['tag']):
        for tag_id in tags:
            try:
                theta = dist.Dirichlet(alpha_q).sample(size)
                pi = dist.Dirichlet(gamma_q[:, u]).sample(size)
                tau = dist.Dirichlet(kappa_q).sample(size)
                sigma = dist.Dirichlet(delta_q[:, tag_id]).sample(size)
                rho = dist.Dirichlet(iota_q[:, tag_id]).sample(size)
                eta  = dist.Dirichlet(zeta_q).sample(size)
                lmd = dist.Multinomial(1, eta).sample()[:,:12,:]

                # lmd: (sample_size, time_count, 2)
                # rho: (sample_size, time_count)
                # sigma: (sample_size, group_count)
                # pi: (sample_size, group_count)
                # tau: (sample_size, group_count, time_size)
                # theta: (sample_size, group_count)
                # aim -> (sample_size, group_count, time_size)
    #             print(lmd.shape)
    #             print(rho.shape)
    #             print(sigma.shape)
    #             sys.exit()

                likelihood_list.append(
                    (
                        ( 
                            (lmd[:,:,0] * rho).unsqueeze(1) 
                      + lmd[:,:,1].unsqueeze(1) * sigma.unsqueeze(2) 
                        )
                     * tau * pi.unsqueeze(2) * theta.unsqueeze(2)
                    ).view(size, -1).sum(1).to('cpu')
                                      )
            except:
                pass
    log_sum_p = torch.sum(torch.log(torch.stack(likelihood_list).to(device))) / size

    perplexity = torch.exp(- log_sum_p / n)

    return perplexity.to('cpu').item()

def calc_word_perplexity_given_user_with_union_model(posterior, test_data, sample_size=100):
    size = torch.LongTensor([sample_size]).to(device)
    n = torch.numel(test_data['tag'])

    alpha_q = posterior['alpha_q'].to(device)
    beta_q = posterior['beta_q'].to(device)
    kappa_q = posterior['kappa_q'].to(device)
    gamma_q = posterior['gamma_q'].to(device)
    delta_q = posterior['delta_q'].to(device)
    epsilon_q = posterior['epsilon_q'].to(device)
    zeta_q = posterior['zeta_q'].to(device)
    iota_q = posterior['iota_q'].to(device)

    likelihood_list = []
    for u, tags in zip(tqdm(test_data['u']), test_data['tag']):
        for tag_id in tags:
            try:
                theta = dist.Dirichlet(alpha_q).sample(size)
                pi = dist.Dirichlet(gamma_q[:, u]).sample(size)
                phi = dist.Dirichlet(beta_q).sample(size)
                tau = dist.Dirichlet(kappa_q).sample(size)
                sigma = dist.Dirichlet(delta_q[:, tag_id]).sample(size)
                mu = dist.Dirichlet(epsilon_q[:, tag_id]).sample(size)
                rho = dist.Dirichlet(iota_q[:, tag_id]).sample(size)
                eta  = dist.Dirichlet(zeta_q).sample(size)
                lmd = dist.Multinomial(1, eta).sample()

                # lmd: (sample_size, location_size, 3)
                # mu: (sample_size, location_size)
                # rho: (sample_size, time_count)
                # sigma: (sample_size, group_count)
                # phi: (sample_size, group_count, location_size)
                # pi: (sample_size, group_count)
                # tau: (sample_size, group_count, time_size)
                # theta: (sample_size, group_count)
                # aim -> (sample_size, group_count, location_size, time_size)
                p_w_given_tlg = (lmd.index_select(2, torch.LongTensor([0]).to(device)) * mu.unsqueeze(2)).unsqueeze(1) \
                    + (lmd.index_select(2, torch.LongTensor([1]).to(device)) * rho.unsqueeze(1)).unsqueeze(1) \
                    + lmd.index_select(2, torch.LongTensor([2]).to(device)).unsqueeze(1) * sigma.view(size, -1, 1, 1)

                likelihood_list.append((p_w_given_tlg * tau.unsqueeze(2) * phi.unsqueeze(3) * (pi * theta).view(size, -1, 1, 1)).view(size, -1).sum(1).to('cpu'))
            except:
                pass

    log_sum_p = torch.sum(torch.log(torch.stack(likelihood_list).to(device))) / size

    perplexity = torch.exp(- log_sum_p / n)

    return perplexity.to('cpu').item()

def calc_location_perplexity_given_user(posterior, test_data, sample_size=100):
    n = torch.numel(test_data['l'])

    alpha_q = posterior['alpha_q'].to(device)
    gamma_q = posterior['gamma_q'].to(device)
    beta_q = posterior['beta_q'].to(device)

    size = torch.LongTensor([sample_size]).to(device)
    likelihood_list = []
    for u, l in zip(tqdm(test_data['u']), test_data['l']):
        try:
            theta = dist.Dirichlet(alpha_q).sample(size)
            pi = dist.Dirichlet(gamma_q[:, u]).sample(size)
            phi = dist.Dirichlet(beta_q[:, l]).sample(size)

            # aim: (sample_size, group_count)
            likelihood_list.append((theta * pi * phi).sum(1).to('cpu'))
        except:
            pass

    log_sum_p = torch.sum(torch.log(torch.stack(likelihood_list).to(device))) / size

    perplexity = torch.exp(- log_sum_p / n)

    return perplexity.to('cpu').item()

def calc_score(posterior, model_type, test_data):
    if model_type not in ['base', 'time', 'location', 'union', 'timeaware']:
        return

    result_metrics = {}

    # calc perplexity
    if model_type in ['base', 'time']:
        w_perplexity_u = calc_word_perplexity_given_user(posterior, test_data, 10)
        print('word_perplexity_given_user:', w_perplexity_u)
        result_metrics.update({'word_perplexity_given_user': w_perplexity_u})
    elif model_type == 'location':
        w_perplexity_u = calc_word_perplexity_given_user_with_location_model(posterior, test_data, 10)
        print('word_perplexity_given_user:', w_perplexity_u)
        result_metrics.update({'word_perplexity_given_user': w_perplexity_u})
    elif model_type == 'timeaware':
        w_perplexity_u = calc_word_perplexity_given_user_with_timeaware_model(posterior, test_data, 10)
        print('word_perplexity_given_user:', w_perplexity_u)
        result_metrics.update({'word_perplexity_given_user': w_perplexity_u})
    elif model_type == 'union':
        w_perplexity_u = calc_word_perplexity_given_user_with_union_model(posterior, test_data, 10)
        print('word_perplexity_given_user:', w_perplexity_u)
        result_metrics.update({'word_perplexity_given_user': w_perplexity_u})
    else:
        print('not model')


    l_perplexity_u = calc_location_perplexity_given_user(posterior, test_data, 10)
    result_metrics.update({'location_perplexity_given_user': l_perplexity_u})
    print('location_perplexity_given_user:', l_perplexity_u)


    return result_metrics

def run(ex):
    eid = ex.split(".")[0]
    # download posterior
#     download_posterior(eid)

    d = torch.load("./pkl_model/" + eid + '.pkl')

    # prepare test data
    test_file = d['data_file'].replace("train","test")
    print(test_file)
    n_test = len(open(test_file).readlines())-1
    test_id_tensor = torch.arange(n_test).to(device)
    test_data = get_test_data(test_file, test_id_tensor)
    model_dict = {"base" : "base",
                  "_s_" : "location",#
                  "_t_" : "timeaware",#timeaware
                  "st"  : "union",
                 }
#     test_data = get_test_data(posterior['data_file'], posterior['test_ids'])

    model_type = model_dict[d["tags"].split(";")[-1]]
    
#     if "split_by_user" in d["tags"] or d["tags"].split(";")[-1] != "_t_":
#         return
    
#     if "split_by_user" in d["tags"] or model_type in ["base","location","union"]:
#         return
    if "split_by_time" in d["tags"]:
#         if os.path.exists(eid + '.pkl'): os.remove(eid + '.pkl')       
        return
    
    print(d["tags"])
#     print(model_type)
    print('start: ', eid, "\n")
    
    metrics = calc_score(d,model_type, test_data)
#     metrics = calc_score(d,"timeaware", test_data)
#     metrics = calc_score(d,"base", test_data)
    
#     data_per_user = divide_data_by_user(test_data, d)

#     # evaluationex.get_tags()[0]
#     metrics = calc_score(d,"base" , data_per_user)

    # log update
#         ex.log_metrics(metrics)

    # rm pickl
#     if os.path.exists(eid + '.pkl'):
#         os.remove(eid + '.pkl')
#     else:
#         print('Could not remove: ', eid + '.pkl')

    print('done: ', eid, "\n")
# except ClientError:
#     print('error: ', eid, "\n")
#         return

def main(args):

    exs = sorted([i for i in os.listdir("./pkl_model") if i.endswith("pkl")])
    n_total = len(exs)
    for i,ex in enumerate(exs):
        print(f' {i+1} / {n_total} ',file = sys.stderr )
        run(ex)

if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.1')
    pyro.enable_validation()
    parser = argparse.ArgumentParser(description='pyro model evaluation')
    parser.add_argument('--debug', action='store_true', help='debug mode')
	
    args = parser.parse_args()
    
    main(args)
