import torch

class Env():
    def __init__(self, device, x, max_retrievals=None):
        super().__init__()
        #x: (batch_size) X (max_stacks) X (max_tiers)
        self.t_pd = 30
        self.t_acc = 40
        self.t_bay = 3.5
        self.t_row = 1.2
        self.device = device
        self.batch, self.n_bays, self.n_rows, self.max_tiers = x.size()
        self.max_stacks = self.n_bays * self.n_rows
        self.x = x.reshape(self.batch, self.max_stacks, self.max_tiers).to(self.device)
        self.target_stack = None
        self.empty = torch.zeros([self.batch], dtype=torch.bool).to(self.device)
        self.retrieved = torch.zeros([self.batch]).to(self.device)
        self.curr_bay = torch.full((self.batch,), -1).to(self.device)
        self.curr_row = torch.full((self.batch,), -1).to(self.device)
        self.relocations = torch.zeros([self.batch]).to(self.device)
        self.retrievals = torch.zeros([self.batch]).to(self.device)
        self.max_retrievals = max_retrievals
        self.early_stopped = torch.zeros([self.batch], dtype=torch.bool).to(self.device)

    def find_target_stack(self):
        mn_val = torch.min(torch.where(self.x == .0, torch.FloatTensor([1+self.max_stacks*self.max_tiers]).to(self.device), self.x), dim=2)[0].to(self.device)
        self.target_stack = torch.argmin(mn_val, dim=1).to(self.device)

    def _update_empty(self):
        bottom_val = self.x[:,:,0].to(self.device) # bottom values
        batch_mx = torch.max(bottom_val, dim=1)[0].to(self.device) # Max (batch)
        self.empty = torch.where(batch_mx>0., False, True).to(self.device) # if batch_mx is 0 => Empty

    def _retrieve_cost(self):
        target_bay = self.target_stack // self.n_rows + 1
        target_row = self.target_stack % self.n_rows + 1
        self.curr_bay = torch.where((self.curr_bay == -1) & self.retrieved, target_bay, self.curr_bay).to(self.device)
        self.curr_row = torch.where((self.curr_row == -1) & self.retrieved, target_row, self.curr_row).to(self.device)

        total_cost = torch.where(
            self.curr_bay != target_bay, 
            self.t_acc + (torch.abs(self.curr_bay - target_bay) * self.t_bay), 
            torch.tensor(0).to(self.device)
        ).to(self.device) + torch.abs(self.curr_row - target_row) * self.t_row
        total_cost = total_cost + torch.abs(target_row - torch.zeros_like(target_row).to(self.device)) * self.t_row + self.t_pd
        total_cost = total_cost * self.retrieved

        self.curr_bay = torch.where(
            self.retrieved,
            target_bay,
            self.curr_bay).to(self.device)
        self.curr_row = torch.where(
            self.retrieved,
            torch.zeros_like(self.curr_row).to(self.device),
            self.curr_row).to(self.device)

        return total_cost
    
    def _relocation_cost(self, source_idx, dest_idx):
        source_idx = source_idx.squeeze(-1)
        dest_idx = dest_idx.squeeze(-1)
        source_bay = source_idx // self.n_rows + 1
        source_row = source_idx % self.n_rows + 1
        dest_bay = dest_idx // self.n_rows + 1
        dest_row = dest_idx % self.n_rows + 1
        self.curr_bay = torch.where(self.curr_bay == -1, source_bay, self.curr_bay).to(self.device)
        self.curr_row = torch.where(self.curr_row == -1, source_row, self.curr_row).to(self.device)

        total_cost = torch.where(
            self.curr_bay != source_bay, 
            self.t_acc + (torch.abs(self.curr_bay - source_bay) * self.t_bay), 
            torch.tensor(0).to(self.device)
        ).to(self.device) + (torch.abs(self.curr_row - source_row) * self.t_row)

        total_cost = total_cost + torch.where(
            source_bay != dest_bay, 
            self.t_acc + (torch.abs(source_bay - dest_bay) * self.t_bay), 
            torch.tensor(0).to(self.device)
        ).to(self.device) + (torch.abs(source_row - dest_row) * self.t_row) + self.t_pd

        total_cost = total_cost * (1.0 - self.empty.type(torch.float64)).to(self.device)

        self.curr_bay = dest_bay
        self.curr_row = dest_row

        return total_cost

    def clear(self):
        # Retrieve
        self.find_target_stack()
        retrieve_cost = torch.tensor([0 for _ in range(self.batch)]).to(self.device)
        # retrieved_blocks = torch.zeros([self.batch]).to(self.device)

        n,s,t = self.batch, self.max_stacks, self.max_tiers
        binary_x = torch.where(self.x > 0., 1, 0).to(self.device) # Block -> 1 Empty -> 0
        stack_len = torch.sum(binary_x, dim=2).to(self.device) # stack length
        target_stack_len = torch.gather(stack_len, dim=1, index = self.target_stack[:,None].to(self.device)).to(self.device) # target_stack height
        stack_mx_index = torch.argmin(torch.where(self.x == .0, torch.FloatTensor([999]).to(self.device), self.x).to(self.device), dim=2).to(self.device)
        target_stack_mx_index = torch.gather(stack_mx_index, dim=1, index=self.target_stack[:,None].to(self.device)).to(self.device)
        clear_mask = ((target_stack_len -1) == target_stack_mx_index).to(self.device)
        clear_mask = (clear_mask & (torch.where(target_stack_len > 0, True, False))).to(self.device) # ignore removed groups
        self.retrieved = clear_mask.squeeze(-1)
        #print('---------------')
        while torch.sum(self.retrieved) > 0:
            #print(clear_mask.squeeze(-1) * self._retrieve_cost())
            retrieve_cost = retrieve_cost + self._retrieve_cost()
            self.retrievals[self.retrieved] += 1

            subtracted_x = self.x - clear_mask.long().view(n,1,1).repeat(1,s,t).to(self.device)
            # retrieved_blocks += self.retrieved.long().to(self.device)
            self.x = torch.where(self.x > 0, subtracted_x, self.x).to(self.device)
            
            # do again
            self.find_target_stack()
            binary_x = torch.where(self.x > 0., 1, 0).to(self.device) # Block -> 1 Empty -> 0
            stack_len = torch.sum(binary_x, dim=2).to(self.device) # stack length
            target_stack_len = torch.gather(stack_len, dim=1, index = self.target_stack[:,None].to(self.device)).to(self.device) # target_stack location
            stack_mx_index = torch.argmin(torch.where(self.x == .0, torch.FloatTensor([999]).to(self.device), self.x).to(self.device), dim=2).to(self.device)
            target_stack_mx_index = torch.gather(stack_mx_index, dim=1, index=self.target_stack[:,None].to(self.device)).to(self.device)
            clear_mask = ((target_stack_len -1) == target_stack_mx_index).to(self.device)
            clear_mask = (clear_mask & (torch.where(target_stack_len > 0, True, False))).to(self.device) # ignore removed groups
            self.retrieved = clear_mask.squeeze(-1)
        self._update_empty()
        # self.last_retrieved_nums = retrieved_blocks

        return retrieve_cost
    
    def step(self, dest_index, source_index=None, no_clear=False):
        if source_index == None:
            source_index = self.target_stack[:, None]
        len_mask = torch.where(self.x > 0., 1, 0).to(self.device)
        stack_len = torch.sum(len_mask, dim=2).to(self.device)
        source_stack_len = torch.gather(stack_len, dim=1, index=source_index).to(self.device)
        dest_stack_len = torch.gather(stack_len, dim=1, index=dest_index).to(self.device)
        top_ind = stack_len - 1
        top_ind = torch.where(top_ind >=0, top_ind, 0).to(self.device)
        top_val = torch.gather(self.x, dim=2, index=top_ind[:,:,None]).to(self.device)
        top_val = top_val.squeeze(-1)
        source_top_val = torch.gather(top_val, dim=1, index=source_index).to(self.device)
        source_ind = source_stack_len - 1
        source_ind = torch.where(source_ind >=0, source_ind, 0).to(self.device)
        input_index = (
            torch.arange(self.batch).to(self.device), 
            source_index.squeeze(-1).to(self.device), 
            source_ind.squeeze(-1).to(self.device)
        )
        self.x = self.x.index_put(input_index, torch.Tensor([0.]).to(self.device)).to(self.device)
        input_index = (
            torch.arange(self.batch).to(self.device), 
            dest_index.squeeze(-1).to(self.device), 
            dest_stack_len.squeeze(-1).to(self.device)
        )
        self.x = self.x.index_put(input_index, source_top_val.squeeze(-1)).to(self.device)

        self.relocations = self.relocations + (1.0 - self.empty.type(torch.float64)).to(self.device)
        total_cost = self._relocation_cost(source_index, dest_index)
        if not no_clear:
            total_cost = total_cost + self.clear()

        return total_cost

    def all_empty(self):
        sum = torch.sum(self.empty.type(torch.int))
        if (sum == self.batch):
            return True
        else:
            return False
    
    def all_terminated(self):
        sum = torch.sum((self.empty | self.early_stopped).type(torch.int))
        if (sum == self.batch):
            return True
        else:
            return False
    
    def set_target_stack(self, target_idx):
        """Override target stack with external selection (HTR mode).

        Args:
            target_idx: (batch,) tensor of stack indices
        """
        self.target_stack = target_idx.to(self.device)

    def create_mask(self):
        top_val = self.x[:,:,-1]
        mask = torch.where(top_val>0, True, False).to(self.device)
        mask = mask.bool()
        target_stack = self.target_stack.clone().to(self.device)
        index = (torch.arange(self.batch).to(self.device), target_stack.squeeze())
        mask = mask.index_put(index, torch.BoolTensor([True]).to(self.device))
        return mask[:,:,None].to(self.device)
