import torch
import torch.nn as nn

class MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_shareoutputlossGate_LossGateUpdate(nn.Module):
    """Five-/multi-node MCP layer used by the Leaf River shared-output model.

    The constructor and return signature are kept compatible with the original
    implementation/checkpoints, while removing unused intermediate variables.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        gate_dim_ucorr: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Kept in the signature for backward compatibility with existing calls.
        # Only hidden_size, batch_first, spinLen, and traintimeLen are used here.
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        # Parameter names/shapes are intentionally unchanged so existing
        # state_dict checkpoints load without modification.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))

        self.relu_l = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        # Preserve the original initialization order/distribution.
        with torch.no_grad():
            self.wo.uniform_(0.0, 1.0)
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)
            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
        del epoch  # Kept only for compatibility with the original interface.

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("This MCP implementation expects seq_length == 1.")

        hidden_size = self.hidden_size
        c_state = x.new_zeros(1, hidden_size)

        # Outputs/diagnostics. Using x.new_zeros keeps dtype/device consistent.
        h_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        l_n = x.new_zeros(batch_size, hidden_size)
        lc_n = x.new_zeros(batch_size, hidden_size)
        bp_n = x.new_zeros(batch_size, hidden_size)  # input bypass is fixed to zero
        gate_ib = x.new_zeros(batch_size, hidden_size)
        gate_oo = x.new_zeros(batch_size, hidden_size)
        gate_ol = x.new_zeros(batch_size, hidden_size)
        gate_ol_constraint = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])
        obs_std = x.new_full((batch_size, hidden_size), obsstd)

        # Constants used by the loss/ET gate scaling in the original model.
        ml = 2.9086
        sl = 1.8980

        bias_b0_yom = self.bias_b0_yom.unsqueeze(0)
        bias_b0_ylm = self.bias_b0_ylm.unsqueeze(0)

        # These fractions depend only on learned parameters, so compute once.
        exp_yom = torch.exp(self.weight_r_yom)
        exp_ylm = torch.exp(self.weight_r_ylm)
        exp_yfm = torch.exp(self.weight_r_yfm)
        gate_sum = exp_yom + exp_ylm + exp_yfm
        oo_scale = exp_yom / gate_sum
        ol_scale = exp_ylm / gate_sum

        for b in range(time_lag, batch_size):
            u1 = x[0, b, 0].reshape(1, 1).expand(-1, hidden_size)
            u2 = x[0, b, 1].reshape(1, 1).expand(-1, hidden_size)

            oo_state = torch.addmm(
                bias_b0_yom,
                (c_state - cmean) / cstd,
                self.weight_b1_yom,
            )
            oo = oo_scale * torch.sigmoid(oo_state)

            ol_state = torch.mm((c_state - cmean) / cstd, self.weight_b1_ylm)
            ol_pet = bias_b0_ylm + ((u2 - ml) / sl) * self.weight_b2_ylm
            ol = ol_scale * torch.sigmoid(ol_state + ol_pet)

            # Preserve the original element-wise PET constraint exactly.
            ol_constraint = x.new_zeros(1, hidden_size)
            for node in range(hidden_size):
                if c_state[0, node] > 0:
                    ol_constraint[0, node] = ol[0, node] - self.relu_l(
                        ol[0, node] - u2[0, node] / c_state[0, node]
                    )
                else:
                    ol_constraint[0, node] = ol[0, node]

            f = 1.0 - oo - ol_constraint

            # Save pre-update fluxes/states, matching the original code.
            h_n[b, :] = (oo * c_state)[0]
            c_n[b, :] = c_state[0]
            l_n[b, :] = (ol * c_state)[0]
            lc_n[b, :] = (ol_constraint * c_state)[0]
            gate_oo[b, :] = oo[0]
            gate_ol[b, :] = ol[0]
            gate_ol_constraint[b, :] = ol_constraint[0]
            gate_f[b, :] = f[0]

            # Input gate/bypass is fixed to zero in this architecture.
            c_state = f * c_state + u1

        # Shared output weights across nodes.
        exp_wo = torch.exp(self.wo)
        output_weights = exp_wo / torch.sum(exp_wo)
        hnout = torch.mm(h_n, output_weights)

        return (
            hnout,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_ib,
            gate_oo,
            gate_ol,
            gate_ol_constraint,
            gate_f,
            obs_std,
        )


class MCPBRNN_Generic_PETconstraint_MultiParallel_outputlossshare_MCA1_InputLinearLayer_LossGateUpdate(nn.Module):
    """Multi-node MCP with learned precipitation distribution across nodes.

    This is the distributed-input variant. The public constructor, parameter
    names/shapes, and return signature are kept compatible with the original
    implementation and existing checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        gate_dim_ucorr: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retain only attributes that are used during forward execution.
        # The full constructor signature is preserved for backward compatibility.
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        # Parameter names/shapes intentionally match the original class so that
        # existing state_dict checkpoints load without modification.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.wi = nn.Parameter(torch.empty(1, hidden_size))

        self.relu_l = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        # Preserve the original initialization order and U(0, 1) distribution.
        with torch.no_grad():
            self.wi.uniform_(0.0, 1.0)
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)
            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
        del epoch  # Kept only for compatibility with the original interface.

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("This MCP implementation expects seq_length == 1.")

        hidden_size = self.hidden_size
        c_state = x.new_zeros(1, hidden_size)

        # Outputs/diagnostics. x.new_zeros preserves input dtype/device.
        h_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        l_n = x.new_zeros(batch_size, hidden_size)
        lc_n = x.new_zeros(batch_size, hidden_size)
        bp_n = x.new_zeros(batch_size, hidden_size)  # input bypass is fixed to zero
        gate_ib = x.new_zeros(batch_size, hidden_size)
        gate_oo = x.new_zeros(batch_size, hidden_size)
        gate_ol = x.new_zeros(batch_size, hidden_size)
        gate_ol_constraint = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])
        obs_std = x.new_ones(batch_size, hidden_size) * obsstd

        # Constants used by the loss/ET gate scaling in the original model.
        ml = 2.9086
        sl = 1.8980

        bias_b0_yom = self.bias_b0_yom.unsqueeze(0)
        bias_b0_ylm = self.bias_b0_ylm.unsqueeze(0)

        # These weights depend only on learned parameters, so compute once.
        exp_wi = torch.exp(self.wi)
        input_weights = exp_wi / torch.sum(exp_wi)

        exp_yom = torch.exp(self.weight_r_yom)
        exp_ylm = torch.exp(self.weight_r_ylm)
        exp_yfm = torch.exp(self.weight_r_yfm)
        gate_sum = exp_yom + exp_ylm + exp_yfm
        oo_scale = exp_yom / gate_sum
        ol_scale = exp_ylm / gate_sum

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1).expand(-1, hidden_size)
            u1 = precipitation * input_weights
            u2 = x[0, b, 1].reshape(1, 1).expand(-1, hidden_size)

            oo_state = torch.addmm(
                bias_b0_yom,
                (c_state - cmean) / cstd,
                self.weight_b1_yom,
            )
            oo = oo_scale * torch.sigmoid(oo_state)

            ol_state = torch.mm((c_state - cmean) / cstd, self.weight_b1_ylm)
            ol_pet = bias_b0_ylm + ((u2 - ml) / sl) * self.weight_b2_ylm
            ol = ol_scale * torch.sigmoid(ol_state + ol_pet)

            # Preserve the original element-wise PET constraint exactly.
            ol_constraint = x.new_zeros(1, hidden_size)
            for node in range(hidden_size):
                if c_state[0, node] > 0:
                    ol_constraint[0, node] = ol[0, node] - self.relu_l(
                        ol[0, node] - u2[0, node] / c_state[0, node]
                    )
                else:
                    ol_constraint[0, node] = ol[0, node]

            f = 1.0 - oo - ol_constraint

            # Save pre-update fluxes/states, matching the original code.
            h_n[b, :] = (oo * c_state)[0]
            c_n[b, :] = c_state[0]
            l_n[b, :] = (ol * c_state)[0]
            lc_n[b, :] = (ol_constraint * c_state)[0]
            gate_oo[b, :] = oo[0]
            gate_ol[b, :] = ol[0]
            gate_ol_constraint[b, :] = ol_constraint[0]
            gate_f[b, :] = f[0]

            # The precipitation input is distributed across nodes by input_weights.
            c_state = f * c_state + u1

        # Total discharge is the sum of node outflows in this architecture.
        hnout = torch.sum(h_n, dim=1).view(batch_size, 1)

        return (
            hnout,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_ib,
            gate_oo,
            gate_ol,
            gate_ol_constraint,
            gate_f,
            obs_std,
        )

class MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate(nn.Module):
    """Implementation of the standard LSTM.
    TODO: Include ref and LaTeX equations
    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))           
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        self.relu_l = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_gw = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):
        """[summary]
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor, containing a batch of input sequences. Format must match the specified format,
            defined by the batch_first agrument.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])       
        h_n = torch.zeros([batch_size, hidden_size])
        gw_n = torch.zeros([batch_size, hidden_size])        
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_oogw = torch.zeros([batch_size, hidden_size])        
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        # MLE sigma
        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_yom_gw = (self.bias_b0_yom_gw.unsqueeze(0).expand(1, *self.bias_b0_yom_gw.size()))        
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)

        mo = p_mean
        ml = 2.9086
        so = p_std
        sl = 1.8980

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                oogw1 = torch.exp(self.weight_r_yom_gw)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oogw2 = torch.addmm(bias_b0_yom_gw, (c_0-mo)/so, self.weight_b1_yom_gw)
                oogw = oogw1 * torch.sigmoid(oogw2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 #u2 * torch.mm(c_0, self.weight_b3_ylm)
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:                
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - oogw - ol_constraint) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g
                h_1 = oo * c_1
                gw_1 = oogw * c_1                
                l_1 = ol * c_1
                bp_0 = ib * g
                h_0 = oo * c_0
                gw_0 = oogw * c_0                
                l_0 = ol * c_0
                lc_0 = ol_constraint * c_0               
            # save state     
                q_n[b,:] = h_0
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0
                gw_n[b,:] = gw_0    

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint 
                Gate_oogw[b,:] = oogw 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd#torch.addmm(torch.exp(self.weight_siga0), torch.exp(self.weight_siga1), ANLLtemp)

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, gw_n, Gate_ib, Gate_oo, Gate_ol, Gate_ol_constraint, Gate_f, Gate_oogw, h_nout, obs_std



class MCPBRNN_GWVariant_Routing(nn.Module):
    """Implementation of the standard LSTM.
    TODO: Include ref and LaTeX equations
    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_GWVariant_Routing, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, Ini_C):
        """[summary]
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor, containing a batch of input sequences. Format must match the specified format,
            defined by the batch_first agrument.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """

        if self.batch_first:
            x = x.transpose(0, 1)

        x = x[:,:, None]
        seq_len, batch_size, _ = x.size()
        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = Ini_C * torch.ones([hidden_size, hidden_size]) 

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
      
        # Gate Function
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))

        mo = c_mean
        so = c_std

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)   

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)            
                f = (1.0 - oo) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                h_0 = oo * c_0                
                bp_0 = ib * g

            # save state     
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0

            # save gate     
                Gate_oo[b,:] = oo
                Gate_f[b,:] = f 
                h_x = (h_1, c_1)
                
        return h_n, c_n, Gate_oo, Gate_f


class MCPBRNN_SW_Variant_Routing(nn.Module):
    """Implementation of the standard LSTM.
    TODO: Include ref and LaTeX equations
    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_SW_Variant_Routing, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))  
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))        
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
    
    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):
        """[summary]
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor, containing a batch of input sequences. Format must match the specified format,
            defined by the batch_first agrument.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        x = x[:,:, None]
        seq_len, batch_size, _ = x.size()
        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
      
        # Gate Function
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))

        mo = p_mean
        so = p_std

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)    

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)
                f = (1.0 - oo) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                h_0 = oo * c_0                
                bp_0 = ib * g

            # save state     
                h_n[b,:] = h_0 
                c_n[b,:] = c_0 # modify from c_1 to c_0

            # save gate     
                Gate_oo[b,:] = oo
                Gate_f[b,:] = f 
                h_x = (h_1, c_1)
                
        return h_n, c_n, Gate_oo, Gate_f


class MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate(nn.Module):
    """Primary MCP storage with surface-flow, groundwater, and ET/loss pathways.

    Parameter names/shapes and the return signature are preserved so existing
    checkpoints remain compatible. This MCA5 implementation is used with one
    storage node (hidden_size=1).
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Keep the public constructor compatible with the original code.
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        # Parameter names/shapes intentionally match the original class.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom_gw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        # Preserve the original U(0, 1) initialization order exactly.
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yom_gw.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)
            self.bias_b0_yom_gw.uniform_(0.0, 1.0)
            self.weight_b1_yom_gw.uniform_(0.0, 1.0)
            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):
        del epoch  # Retained only for interface compatibility.

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("This MCA5 implementation expects seq_length == 1.")

        hidden_size = self.hidden_size
        c_state = x.new_zeros(1, hidden_size)

        # Time-series diagnostics/fluxes.
        h_n = x.new_zeros(batch_size, hidden_size)
        gw_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        l_n = x.new_zeros(batch_size, hidden_size)
        lc_n = x.new_zeros(batch_size, hidden_size)
        bp_n = x.new_zeros(batch_size, hidden_size)

        gate_ib = x.new_zeros(batch_size, hidden_size)
        gate_oo = x.new_zeros(batch_size, hidden_size)
        gate_oogw = x.new_zeros(batch_size, hidden_size)
        gate_ol = x.new_zeros(batch_size, hidden_size)
        gate_ol_constraint = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])
        obs_std = x.new_ones(batch_size, hidden_size) * obsstd

        # PET normalization constants used in the original model.
        pet_mean = 2.9086
        pet_std = 1.8980

        bias_b0_yom = self.bias_b0_yom.unsqueeze(0)
        bias_b0_yom_gw = self.bias_b0_yom_gw.unsqueeze(0)
        bias_b0_ylm = self.bias_b0_ylm.unsqueeze(0)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1).expand(-1, hidden_size)
            pet = x[0, b, 1].reshape(1, 1).expand(-1, hidden_size)

            surface_state = torch.addmm(
                bias_b0_yom,
                (c_state - p_mean) / p_std,
                self.weight_b1_yom,
            )
            surface_scale = torch.exp(self.weight_r_yom) / (
                torch.exp(self.weight_r_yom)
                + torch.exp(self.weight_r_yom_gw)
                + torch.exp(self.weight_r_ylm)
                + torch.exp(self.weight_r_yfm)
            )
            gate_surface = surface_scale * torch.sigmoid(surface_state)

            groundwater_state = torch.addmm(
                bias_b0_yom_gw,
                (c_state - p_mean) / p_std,
                self.weight_b1_yom_gw,
            )
            groundwater_scale = torch.exp(self.weight_r_yom_gw) / (
                torch.exp(self.weight_r_yom)
                + torch.exp(self.weight_r_yom_gw)
                + torch.exp(self.weight_r_ylm)
                + torch.exp(self.weight_r_yfm)
            )
            gate_groundwater = groundwater_scale * torch.sigmoid(groundwater_state)

            loss_pet = torch.addmm(
                bias_b0_ylm,
                (pet - pet_mean) / pet_std,
                self.weight_b2_ylm,
            )
            loss_scale = torch.exp(self.weight_r_ylm) / (
                torch.exp(self.weight_r_yom)
                + torch.exp(self.weight_r_yom_gw)
                + torch.exp(self.weight_r_ylm)
                + torch.exp(self.weight_r_yfm)
            )
            gate_loss = loss_scale * torch.sigmoid(loss_pet)

            # Preserve the original PET constraint node by node.
            constrained_loss = x.new_zeros(1, hidden_size)
            for node in range(hidden_size):
                if c_state[0, node] > 0:
                    constrained_loss[0, node] = gate_loss[0, node] - self.relu_l(
                        gate_loss[0, node] - pet[0, node] / c_state[0, node]
                    )
                else:
                    constrained_loss[0, node] = gate_loss[0, node]

            gate_retention = 1.0 - gate_surface - gate_groundwater - constrained_loss

            # Save pre-update states/fluxes, matching the original implementation.
            h_n[b, :] = (gate_surface * c_state)[0]
            gw_n[b, :] = (gate_groundwater * c_state)[0]
            c_n[b, :] = c_state[0]
            l_n[b, :] = (gate_loss * c_state)[0]
            lc_n[b, :] = (constrained_loss * c_state)[0]
            gate_oo[b, :] = gate_surface[0]
            gate_oogw[b, :] = gate_groundwater[0]
            gate_ol[b, :] = gate_loss[0]
            gate_ol_constraint[b, :] = constrained_loss[0]
            gate_f[b, :] = gate_retention[0]

            # Input bypass is fixed to zero, so all precipitation enters storage.
            c_state = gate_retention * c_state + precipitation

        # This diagnostic return is retained for backward compatibility.
        h_nout = torch.cat((h_n, obs_std), dim=1)

        return (
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gw_n,
            gate_ib,
            gate_oo,
            gate_ol,
            gate_ol_constraint,
            gate_f,
            gate_oogw,
            h_nout,
            obs_std,
        )


class MCPBRNN_GWVariant_Routing(nn.Module):
    """Single-node groundwater-routing MCP used by the MCA5 architecture."""

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Preserve the original U(0, 1) initialization order exactly.
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, Ini_C):
        del epoch, y_obs  # Retained only for interface compatibility.

        if self.batch_first:
            x = x.transpose(0, 1)
        x = x[:, :, None]

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("This groundwater-routing implementation expects seq_length == 1.")

        hidden_size = self.hidden_size
        c_state = x.new_ones(1, hidden_size) * Ini_C

        h_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        gate_oo = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)

        bias_b0_yom = self.bias_b0_yom.unsqueeze(0)
        for b in range(time_lag, batch_size):
            recharge = x[0, b, 0].reshape(1, 1).expand(-1, hidden_size)

            output_state = torch.addmm(
                bias_b0_yom,
                (c_state - c_mean) / c_std,
                self.weight_b1_yom,
            )
            output_scale = torch.exp(self.weight_r_yom) / (
                torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yfm)
            )
            gate_output = output_scale * torch.sigmoid(output_state)
            gate_retention = 1.0 - gate_output

            # Save pre-update state/flux, matching the original implementation.
            h_n[b, :] = (gate_output * c_state)[0]
            c_n[b, :] = c_state[0]
            gate_oo[b, :] = gate_output[0]
            gate_f[b, :] = gate_retention[0]

            c_state = gate_retention * c_state + recharge

        return h_n, c_n, gate_oo, gate_f


class MCPBRNN_SW_Variant_Routing(nn.Module):
    """Single-node surface-water routing MCP used by the MCA5 architecture."""

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Preserve the original U(0, 1) initialization order exactly.
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):
        del epoch, y_obs  # Retained only for interface compatibility.

        if self.batch_first:
            x = x.transpose(0, 1)
        x = x[:, :, None]

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("This surface-routing implementation expects seq_length == 1.")

        hidden_size = self.hidden_size
        c_state = x.new_zeros(1, hidden_size)

        h_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        gate_oo = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)

        bias_b0_yom = self.bias_b0_yom.unsqueeze(0)
        for b in range(time_lag, batch_size):
            inflow = x[0, b, 0].reshape(1, 1).expand(-1, hidden_size)

            output_state = torch.addmm(
                bias_b0_yom,
                (c_state - p_mean) / p_std,
                self.weight_b1_yom,
            )
            output_scale = torch.exp(self.weight_r_yom) / (
                torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yfm)
            )
            gate_output = output_scale * torch.sigmoid(output_state)
            gate_retention = 1.0 - gate_output

            # Save pre-update state/flux, matching the original implementation.
            h_n[b, :] = (gate_output * c_state)[0]
            c_n[b, :] = c_state[0]
            gate_oo[b, :] = gate_output[0]
            gate_f[b, :] = gate_retention[0]

            c_state = gate_retention * c_state + inflow

        return h_n, c_n, gate_oo, gate_f

class MCPBRNN_Generic_PETconstraint_Scaling(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Scaling, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        self.relu_l = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
 
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])        
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_olc = torch.zeros([batch_size, hidden_size])        
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = cmean
        ml = 2.9086
        so = cstd
        sl = 1.8980        
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

                ib = 0 

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint)

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                lc_1 = ol_constraint  * c_1                
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 
                lc_0 = ol_constraint  * c_0 
            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_olc[b,:] = ol_constraint              
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd 

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_olc, Gate_f, h_nout, obs_std
        
class MCPBRNN_Generic_PETconstraint_Scaling(nn.Module):
    """Single-node MCP used by the Leaf River scaling experiment.

    The public constructor, parameter names/shapes, and return signature are
    preserved so existing state_dict checkpoints remain compatible.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained for checkpoint/call compatibility.
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        # Parameter names and shapes intentionally match the original class.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        # Preserve the original initialization order and U(0, 1) draws.
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)
            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
        del epoch  # Kept only for compatibility with the original interface.

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("This MCP implementation expects seq_length == 1.")
        if self.hidden_size != 1:
            raise ValueError("MCPBRNN_Generic_PETconstraint_Scaling is a single-node model.")

        hidden_size = self.hidden_size
        c_state = x.new_zeros(1, hidden_size)

        # Time-series outputs/diagnostics.
        h_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        l_n = x.new_zeros(batch_size, hidden_size)
        lc_n = x.new_zeros(batch_size, hidden_size)
        bp_n = x.new_zeros(batch_size, hidden_size)
        gate_i = x.new_zeros(batch_size, hidden_size)
        gate_o = x.new_zeros(batch_size, hidden_size)
        gate_l = x.new_zeros(batch_size, hidden_size)
        gate_lc = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])
        obs_std = x.new_ones(batch_size, hidden_size) * obsstd

        # Constants used in the original PET-loss gate scaling.
        ml = 2.9086
        sl = 1.8980

        bias_b0_yom = self.bias_b0_yom.unsqueeze(0)
        bias_b0_ylm = self.bias_b0_ylm.unsqueeze(0)

        # These fractions depend only on learned parameters, so compute once.
        exp_yom = torch.exp(self.weight_r_yom)
        exp_ylm = torch.exp(self.weight_r_ylm)
        exp_yfm = torch.exp(self.weight_r_yfm)
        gate_sum = exp_yom + exp_ylm + exp_yfm
        oo_scale = exp_yom / gate_sum
        ol_scale = exp_ylm / gate_sum

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            oo_state = torch.addmm(
                bias_b0_yom,
                (c_state - cmean) / cstd,
                self.weight_b1_yom,
            )
            oo = oo_scale * torch.sigmoid(oo_state)

            ol_pet = torch.addmm(
                bias_b0_ylm,
                (pet - ml) / sl,
                self.weight_b2_ylm,
            )
            ol = ol_scale * torch.sigmoid(ol_pet)

            # Preserve the original single-node PET constraint.
            if c_state.item() > 0:
                ol_constraint = ol - self.relu_l(ol - pet / c_state)
            else:
                ol_constraint = ol

            f = 1.0 - oo - ol_constraint

            # Save pre-update fluxes/states, matching the original implementation.
            h_n[b, :] = (oo * c_state)[0]
            c_n[b, :] = c_state[0]
            l_n[b, :] = (ol * c_state)[0]
            lc_n[b, :] = (ol_constraint * c_state)[0]
            gate_o[b, :] = oo[0]
            gate_l[b, :] = ol[0]
            gate_lc[b, :] = ol_constraint[0]
            gate_f[b, :] = f[0]

            # Input gate/bypass is fixed to zero.
            c_state = f * c_state + precipitation

        h_nout = torch.cat((h_n, obs_std), dim=1)

        return (
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_f,
            h_nout,
            obs_std,
        )

class MCPBRNN_PETconstraint_MassRelax_Regular(nn.Module):
    """Single-node MCP with state-dependent mass relaxation.

    The public constructor, parameter names/shapes, and return signature are
    preserved so existing state_dict checkpoints remain compatible.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        gate_dim_ucorr: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained for call/checkpoint compatibility.
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        # Parameter names and effective post-reset shapes match the original.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.empty(hidden_size))

        self.relu_l = nn.ReLU()
        self.relu_v = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        # Preserve original initialization order and U(0, 1) draws.
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.weight_r_yvm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)
            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)
            self.weight_s_yvm.uniform_(0.0, 1.0)
            self.bias_b0_yrm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):
        del epoch  # Kept only for interface compatibility.

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("This MCP implementation expects seq_length == 1.")
        if self.hidden_size != 1:
            raise ValueError("MCPBRNN_PETconstraint_MassRelax_Regular is a single-node model.")

        hidden_size = self.hidden_size
        c_state = x.new_zeros(1, hidden_size)

        # Time-series outputs/diagnostics.
        l_n = x.new_zeros(batch_size, hidden_size)
        lc_n = x.new_zeros(batch_size, hidden_size)
        h_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        bp_n = x.new_zeros(batch_size, hidden_size)

        gate_i = x.new_zeros(batch_size, hidden_size)
        gate_o = x.new_zeros(batch_size, hidden_size)
        gate_l = x.new_zeros(batch_size, hidden_size)
        gate_ov = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)
        gate_lc = x.new_zeros(batch_size, hidden_size)

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])
        obs_std = x.new_ones(batch_size, hidden_size) * obsstd

        # Original scaling constants.
        ml = 2.9086
        sl = 1.8980
        scale_mr = 500.0

        bias_b0_yom = self.bias_b0_yom.unsqueeze(0)
        bias_b0_ylm = self.bias_b0_ylm.unsqueeze(0)
        bias_b0_yrm = self.bias_b0_yrm.unsqueeze(0)

        # Fractions that depend only on learned parameters.
        exp_yom = torch.exp(self.weight_r_yom)
        exp_ylm = torch.exp(self.weight_r_ylm)
        exp_yfm = torch.exp(self.weight_r_yfm)
        gate_sum = exp_yom + exp_ylm + exp_yfm

        oo_scale = exp_yom / gate_sum
        ol_scale = exp_ylm / gate_sum
        mr_scale = torch.sigmoid(self.weight_r_yvm)
        mr_slope = torch.exp(self.weight_s_yvm)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            # Discharge gate.
            oo_state = torch.addmm(
                bias_b0_yom,
                (c_state - p_mean) / p_std,
                self.weight_b1_yom,
            )
            oo = oo_scale * torch.sigmoid(oo_state)

            # PET/loss gate.
            ol_pet = torch.addmm(
                bias_b0_ylm,
                (pet - ml) / sl,
                self.weight_b2_ylm,
            )
            ol = ol_scale * torch.sigmoid(ol_pet)

            # Preserve the original PET constraint exactly.
            if c_state.item() > 0:
                ol_constraint = ol - self.relu_l(ol - pet / c_state)
            else:
                ol_constraint = ol

            f = 1.0 - oo - ol_constraint

            # Mass-relaxation gate.
            ov0 = torch.mm(
                c_state / scale_mr - bias_b0_yrm,
                mr_slope,
            )
            ov1 = mr_scale * torch.tanh(ov0)
            ov = ov1 - self.relu_v(ov1 - f)

            # Save pre-update states/fluxes, matching the original code.
            h_n[b, :] = (oo * c_state)[0]
            c_n[b, :] = c_state[0]
            l_n[b, :] = (ol * c_state)[0]
            lc_n[b, :] = (ol_constraint * c_state)[0]

            gate_o[b, :] = oo[0]
            gate_l[b, :] = ol[0]
            gate_ov[b, :] = ov[0]
            gate_f[b, :] = f[0]
            gate_lc[b, :] = ol_constraint[0]

            # Input gate/bypass is fixed to zero.
            c_state = (
                f * c_state
                + precipitation
                - ov * torch.abs(c_state - bias_b0_yrm * scale_mr)
            )

        h_nout = torch.cat((h_n, obs_std), dim=1)

        return (
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_f,
            h_nout,
            obs_std,
            gate_ov,
        )