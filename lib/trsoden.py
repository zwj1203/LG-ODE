import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb
class Gradient(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.autograd.grad(y, x)[0]

def append_activation(activation, layers: list):
    if activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise NotImplementedError

class ODEFunc(nn.Module):
    def __init__(self, dim=1, time_augment=False,
                 nb_units=1000, nb_layers=1, activation='tanh'):
        super(ODEFunc, self).__init__()
        self.time_augment = time_augment
        layers = []
        if time_augment:
            layers.append(nn.Linear(2 * dim + 1, nb_units))
        else:
            layers.append(nn.Linear(2 * dim, nb_units))

        append_activation(activation, layers)
        for _ in range(nb_layers - 1):
            layers.append(nn.Linear(nb_units, nb_units))
            append_activation(activation, layers)
        layers.append(nn.Linear(nb_units, 2 * dim, bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, t):
        '''
        inputs: x or [x, t]
        '''
        if self.time_augment:
            inputs = torch.cat([x, t], dim=1)
        else:
            inputs = x
        return self.layers(inputs)

class HamiltionEquation(nn.Module):
    def __init__(self, dim=1, time_augment=False,
                 nb_units=1000, nb_layers=1, activation='tanh'):
        super(HamiltionEquation, self).__init__()
        self.time_augment = time_augment
        self.dim = dim

        q_layers = []
        p_layers = []
        if self.time_augment:
            q_layers.append(nn.Linear(self.dim + 1, self.units))
            p_layers.append(nn.Linear(self.dim + 1, self.units))
        else:
            q_layers.append(nn.Linear(self.dim, self.units))
            p_layers.append(nn.Linear(self.dim, self.units))
        self.append_activation(q_layers)
        self.append_activation(p_layers)
        for _ in range(self.layers - 1):
            q_layers.append(nn.Linear(self.units, self.units))
            p_layers.append(nn.Linear(self.units, self.units))
            self.append_activation(q_layers)
            self.append_activation(p_layers)
        
        q_layers.append(nn.Linear(self.units, 1, bias=False))
        p_layers.append(nn.Linear(self.units, 1, bias=False))
        
        self.q_layers, self.p_layers = nn.Sequential(*q_layers), nn.Sequential(*p_layers)

    def forward(self, x, t):
        '''
        inputs: x or [x, t]
        '''
    
        if self.time_augment:
            q_inputs = torch.cat([x[:self.dim], t], dim=1)
            p_inputs = torch.cat([x[self.dim:], t], dim=1)
        else:
            q_inputs = x[:self.dim]
            p_inputs = x[self.dim:]

        v = self.q_layers(q_inputs)
        k = self.p_layers(p_inputs)
        dq = torch.autograd.grad(k, p, create_graph=True)[0]
        dp = - torch.autograd.grad(v, q, create_graph=True)[0]

        return torch.cat([dq, dp], dim=1)

class ODENetwork(nn.Module):
    def __init__(self, nb_object=1, nb_coords=2, function_type='ode', time_augment=False,
                 nb_units=1000, nb_layers=1, activation='tanh', with_gnn=False,
                 lambda_trs=0.0, learning_rate=2e-4):
        super(ODENetwork, self).__init__()
        self.dim = int(nb_object * nb_coords)
        self.augment = time_augment
        self.nb_object = nb_object
        self.nb_coords = nb_coords
        
        self.lambda_trs = lambda_trs
        self.lr = learning_rate
        self.with_gnn = with_gnn
        
        if function_type == 'ode':
            self.func = ODEFunc(self.dim , time_augment, nb_units, nb_layers, activation)
        elif function_type == 'hamiltonian':
            self.func = HamiltionEquation(self.dim , time_augment, nb_units, nb_layers, activation)
        else:
            raise NotImplementedError


    def solve(self, ts, x0, padding=False):
        '''
        ts: batch_de["time_steps"], shape [n_timepoints] 
        x0: expected [batchsize, n x spatial dim x 2]
        '''
        if ts[0] != 0:
            ts = torch.cat([torch.zeros(1), ts])
        dts = ts[1:] - ts[:-1]
        x = x0
        xr = torch.cat([x0[:, :self.dim], -x0[:, self.dim:]], dim=1)

        ls_x, ls_xr = [], []

        for i in range(ts.shape[0]-1):
            t = ts[i].repeat(x.shape[0], 1)
            tr = -t
            dt = dts[i]

            if self.augment:
                dx1 = self.func(x, t) * dt
                dx2 = self.func(x + 0.5 * dx1, t + 0.5 * dt) * dt
                dx3 = self.func(x + 0.5 * dx2, t + 0.5 * dt) * dt
                dx4 = self.func(x + dx3, t + dt) * dt
                dx = (1 / 6) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
                x = x + dx
                ls_x.append(x)

                dxr1 = - self.func(xr, tr) * dt
                dxr2 = - self.func(xr + 0.5 * dxr1, tr - 0.5 * dt) * dt
                dxr3 = - self.func(xr + 0.5 * dxr2, tr - 0.5 * dt) * dt
                dxr4 = - self.func(xr + dxr3, tr - dt) * dt
                dxr = (1 / 6) * (dxr1 + 2 * dxr2 + 2 * dxr3 + dxr4)
                xr = xr + dxr
                ls_xr.append(xr)
   
            else:  # Leapfrog solver for autonomous systems
                # Forward time evolution
                # pdb.set_trace()
                q, p = x[:, :self.dim], x[:, self.dim:]
                p = p + 0.5 * dt * self.func(x, t)[:, self.dim:] 
                q = q + 1.0 * dt * self.func(torch.cat([q, p], dim=-1), t)[:, :self.dim] 
                p = p + 0.5 * dt * self.func(torch.cat([q, p], dim=-1), t)[:, self.dim:] 
                x = torch.cat([q, p], dim=-1)
                ls_x.append(x)

                # Backward time evolution
                qr, pr = xr[:, :self.dim], xr[:, self.dim:]
                pr = pr - 0.5 * dt * self.func(xr, t)[:, self.dim:]
                qr = qr - 1.0 * dt * self.func(torch.cat([qr, pr], dim=-1), t)[:, :self.dim] 
                pr = pr - 0.5 * dt * self.func(torch.cat([qr, pr], dim=-1), t)[:, self.dim:] 
                xr = torch.cat([qr, pr], dim=-1)
                ls_xr.append(xr)

        return torch.stack(ls_x, dim=1), torch.stack(ls_xr, dim=1)

    def first_point_imputation(self, batch_enc, batch_dec):
        '''
        batch_enc["data"] : [b x n_objects, D]
        batch_enc["time_steps"] : [b x n_objects]

        batch_dec["data"] : [b x n_objects, T, D]
        batch_dec["time_steps"] : [b x n_objects, T]
        '''
        dec_indices_orig_init_value = (batch_dec["time_first"] == 0).nonzero(as_tuple=False).squeeze()
        # pdb.set_trace()
        time_intervals = (batch_dec["time_first"] - batch_enc["time_steps"]).unsqueeze(1)
        computed_init_states = batch_enc["data"]*batch_dec["time_first"].unsqueeze(1)/time_intervals - batch_dec["data"][:,0,:]*batch_enc["time_steps"].unsqueeze(1)/time_intervals 
        computed_init_states[dec_indices_orig_init_value] = batch_dec["data"][dec_indices_orig_init_value, 0, :] 

        return computed_init_states

    def compute_loss(self, batch_enc, batch_dec):
        init_states = self.first_point_imputation(batch_enc, batch_dec) #b x n_objects, D
        b = init_states.shape[0] // self.nb_object

        if not self.with_gnn:
            q = init_states[:, :self.nb_coords].reshape(b, self.nb_object, -1).reshape(b, -1) # [b, N x D//2]
            p = init_states[:, self.nb_coords:].reshape(b, self.nb_object, -1).reshape(b, -1)
            
            x0 = torch.cat([q, p], dim=-1)

            ts = batch_dec["time_steps"]
            padding = False
            if ts[0] != 0:
                ts = torch.cat([torch.zeros(1), ts])
                padding = True
            
            X, Xr = self.solve(ts, x0, padding) # [B, T, N x d x 2]

            T, D = X.shape[1], X.shape[2] // 2
            Xq, Xp = X[:,:,:D], X[:,:,D:]
            Xrq, Xrp = Xr[:,:,:D], Xr[:,:,D:]
            
            Xq = Xq.reshape(b, T, self.nb_object, self.nb_coords).permute(0, 2, 1, 3).reshape(-1, T, self.nb_coords)
            Xp = Xp.reshape(b, T, self.nb_object, self.nb_coords).permute(0, 2, 1, 3).reshape(-1, T, self.nb_coords)
            Xrq = Xrq.reshape(b, T, self.nb_object, self.nb_coords).permute(0, 2, 1, 3).reshape(-1, T, self.nb_coords)
            Xrp = Xrp.reshape(b, T, self.nb_object, self.nb_coords).permute(0, 2, 1, 3).reshape(-1, T, self.nb_coords)
            
            if padding:
                mask = batch_dec["mask"]
            else:
                mask = batch_dec["mask"][:, 1:, :]
                batch_dec["data"] = batch_dec["data"][:, 1:, :]
             
            # X = torch.cat([Xq, Xp], dim=2)
            timelength_per_nodes = torch.sum(mask.permute(0,2,1),dim=2)

            # mask = mask.reshape(b, self.nb_object, T, -1).permute(0,2,1,3).reshape(b, T, -1)
            # pdb.set_trace()
            forward_diff = torch.square(torch.cat([Xq, Xp], dim=2) - batch_dec["data"]) * mask
            forward_diff = forward_diff.sum(dim=1) / timelength_per_nodes
            l_ode = torch.mean(forward_diff)

            fr_diff = torch.square(torch.cat([Xq, -Xp], dim=2) - torch.cat([Xrq, Xrp], dim=2)) * mask
            fr_diff = fr_diff.sum(dim=1) / timelength_per_nodes
            l_trs = torch.mean(fr_diff)

            l = l_ode + self.lambda_trs * l_trs
            
            #sum over time dim
            forward_mape = torch.sum(torch.abs(torch.cat([Xq, Xp], dim=2) - batch_dec["data"]) * mask, dim=1)
            # pdb.set_trace()
            forward_mape = forward_mape / torch.sum(torch.abs(batch_dec["data"]) * mask, dim=1)
            forward_mape = forward_mape / timelength_per_nodes
            forward_mape = torch.mean(forward_mape)
        else:
            raise NotImplementedError

        results = {}
        results["loss"] = l
        results["mse"] = l_ode.data.item() + l_trs.data.item()
        results["forward_gt_mse"] = l_ode.data.item()
        results["reverse_f_mse"] = l_trs.data.item()
        results["mape"] = forward_mape.data.item()

        return results

        





        
