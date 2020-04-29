# Define the model
class freyface_model(nn.Module):
    def __init__(self):
        super(freyface_model, self).__init__()

        self.fc1 = nn.Linear(560, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 20)
        self.fc32 = nn.Linear(200, 20)

        self.fc4 = nn.Linear(20, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 560)
        self.fc7 = nn.Linear(200, 560)

        self.K = K

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = F.softplus(self.fc1(x))
        h2 = F.softplus(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z,test=False):
        #h3 = F.relu(self.fc3(z))
        h3 = F.softplus(self.fc4(z))
        h4 = F.softplus(self.fc5(h3))
        return self.fc6(h4), self.fc7(h4) #mu, log_sigma

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, 560))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_loss_for_batch(self, data, model, K=K,test=False):
        # data = (N,560)
        if model_type=='vae':
            alpha=1
        elif model_type in ('iwae','vrmax'):
            alpha=0
        else:
            # use whatever alpha is defined in hyperparameters
            if abs(alpha-1)<=1e-3:
                alpha=1

        data_k_vec = data.repeat_interleave(K,0)

        mu, logstd = model.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, logstd)

        # summing over latents due to independence assumption
        # (B*K)
        log_q = compute_log_probabitility_gaussian(z, mu, logstd)

        log_p_z = torch.sum(-0.5 * z ** 2, 1)-.5*z.shape[1]*T.log(torch.tensor(2*np.pi)) 
        decoded = model.decode(z)
        (pmu, plog_sigma) = decoded
        log_p = compute_log_probabitility_gaussian(data_k_vec,pmu, plog_sigma)
        # hopefully this reshape operation magically works like always
        if model_type == 'iwae' or test==True:
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K)
        elif model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, 1)*1/K
        elif model_type=='general_alpha':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1-alpha)
        elif model_type=='vrmax':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).max(axis=1,keepdim=True).values

        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded, loss