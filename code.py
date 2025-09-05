
class IndexedChessDataset(Dataset):
    def __init__(self, h5_path, max_game_length=None):
        """
        Dataset that loads individual games from indexed format
        
        Args:
            h5_path: Path to indexed H5 file
            max_game_length: Optional limit on game length (for memory management)
        """
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.max_game_length = max_game_length
        
        # Load game boundaries
        self.game_starts = self.h5_file['game_starts'][:]
        self.game_ends = self.h5_file['game_ends'][:]
        self.winner_token_id = 0
        self.loser_token_id = 1
        
        # Filter games by length if specified
        if max_game_length:
            game_lengths = self.game_ends - self.game_starts
            valid_games = game_lengths <= max_game_length
            self.game_starts = self.game_starts[valid_games]
            self.game_ends = self.game_ends[valid_games]
            print(f"Filtered to {len(self.game_starts):,} games with length <= {max_game_length}")
        
        # Get data keys (excluding our index datasets)
        self.data_keys = [key for key in self.h5_file.keys() 
                         if key not in ['game_starts', 'game_ends']]
        
        print(f"Dataset initialized with {len(self.game_starts):,} games")
    
    def __len__(self):
        return len(self.game_starts)
    
    def __getitem__(self, idx):
        start_pos = self.game_starts[idx]
        end_pos = self.game_ends[idx]
        # actual_batch_size = end_pos - start_pos 
        
        # Load game data for all keys
        game_data = {}
        for key in self.data_keys:
            data = self.h5_file[key][start_pos:end_pos]
            game_data[key] = torch.tensor(data, dtype=torch.long)
            
            # # Handle different data types appropriately
            # if 'fen_arrays' in key:
            #     # FEN arrays should be float for neural networks
            #     game_data[key] = torch.tensor(data, dtype=torch.float32)
            # else:
            #     # Other data (moves, flags, etc.) should be long
            #     game_data[key] = torch.tensor(data, dtype=torch.long)
        
        # Create input/target pairs for language modeling
        # Input: all moves except last, Target: all moves except first
        inputs = {}
        targets = {}
        
        for key in self.data_keys:
            if len(game_data[key]) > 1:
                inputs[key] = game_data[key][:-1]  # All but last
                targets[key] = game_data[key][1:]   # All but first
            else:
                # Handle single-move games
                inputs[key] = game_data[key]
                targets[key] = game_data[key]
        # Load entire batch at once (much more efficient)
        batch = {
            'winner_move_froms': inputs['winner_move_froms'],
            'winner_move_tos': inputs['winner_move_tos'],
            # 'winner_pieces': inputs['winner_pieces'],
            'winner_winners': inputs['winner_winners'],
            'winner_turn_list': inputs['winner_turn_list'],
            # 'winner_game_ids': [g.decode('ascii') for g in inputs['winner_guids'][start:end]],#.squeeze(0),
            'winner_fen_arrays': inputs['winner_fen_arrays'],
            'winner_is_en_passant_list': inputs['winner_is_en_passant_list'],
            'winner_is_castling_list': inputs['winner_is_castling_list'],
            
            'loser_move_froms': inputs['loser_move_froms'],
            'loser_move_tos': inputs['loser_move_tos'],
            # 'loser_pieces': inputs['loser_pieces'],
            'loser_winners': inputs['loser_winners'],
            'loser_turn_list': inputs['loser_turn_list'],
            # 'loser_game_ids': [g.decode('ascii') for g in inputs['loser_guids'][start:end]],#.squeeze(0),
            'loser_fen_arrays': inputs['loser_fen_arrays'],
            'loser_is_en_passant_list': inputs['loser_is_en_passant_list'],
            'loser_is_castling_list': inputs['loser_is_castling_list'],
        }
        # --- CRITICAL: CREATE THE CONTEXT-AWARE INPUT ---
        # Get the batch size

        
        # Create a tensor for the winner perspective token.
        # We'll add a new dimension to make it [batch_size, 1] and then project it later.
        # To this:
        actual_sequence_length = end_pos - start_pos
        input_sequence_length = actual_sequence_length - 1  # Match the input/target length
        winner_perspective_tokens = torch.full((input_sequence_length,), self.winner_token_id, dtype=torch.long)
        loser_perspective_tokens = torch.full((input_sequence_length,), self.loser_token_id, dtype=torch.long)

        batch['winner_perspective_token'] = winner_perspective_tokens
        batch['loser_perspective_token'] = loser_perspective_tokens
        
        target_batch= {
            'winner_move_froms': targets['winner_move_froms'],
            'winner_move_tos': targets['winner_move_tos'],
            'winner_winners': targets['winner_winners'],
            'loser_winners': targets['loser_winners'],
            'loser_move_froms': targets['loser_move_froms'],
            'loser_move_tos': targets['loser_move_tos'],
        }
        # Since all have same shape, we can use torch.where directly
        white_move_froms = torch.where(target_batch['winner_winners'] == 0, target_batch['winner_move_froms'], target_batch['loser_move_froms'])
        black_move_froms = torch.where(target_batch['winner_winners'] == 1, target_batch['winner_move_froms'], target_batch['loser_move_froms'])
        white_move_tos = torch.where(target_batch['winner_winners'] == 0, target_batch['winner_move_tos'], target_batch['loser_move_tos'])
        black_move_tos = torch.where(target_batch['winner_winners'] == 1, target_batch['winner_move_tos'], target_batch['loser_move_tos'])

        # Now convert moves >= (length - 3) to -100 for ignoring
        white_move_froms = torch.where(white_move_froms >= from_len - 3,
                                      torch.tensor(-100, device=white_move_froms.device),
                                      white_move_froms)
        
        black_move_froms = torch.where(black_move_froms >= from_len - 3,
                                      torch.tensor(-100, device=black_move_froms.device),
                                      black_move_froms)
        
        white_move_tos = torch.where(white_move_tos >= to_len - 3,
                                    torch.tensor(-100, device=white_move_tos.device),
                                    white_move_tos)
        
        black_move_tos = torch.where(black_move_tos >= to_len - 3,
                                    torch.tensor(-100, device=black_move_tos.device),
                                    black_move_tos)


        targets = {'white_move_froms' : white_move_froms, 'black_move_froms':black_move_froms, 
                   'white_move_tos':white_move_tos,  'black_move_tos': black_move_tos}
        
        msg = f"Input and target batch size mismatch: {batch['winner_fen_arrays'].size(0)} vs {target_batch['winner_move_tos'].size(0)}"
        # Check if the batch sizes align
        assert batch['winner_fen_arrays'].size(0) == target_batch['winner_move_tos'].size(0), msg 
        
        # Separate moves into winners and losers
        return batch, targets
        
        # return inputs, targets
    
    def get_game_length(self, idx):
        """Get length of specific game"""
        return self.game_ends[idx] - self.game_starts[idx]
    
    def get_game_lengths(self):
        """Get lengths of all games"""
        return self.game_ends - self.game_starts
    
    def close(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass
    
    def __del__(self):
        self.close()
vocab_sizes = {
            'winner_move_froms': from_len,
            'winner_move_tos': to_len,
            # 'winner_pieces': inputs['winner_pieces'],
            'winner_winners': 2,
            'winner_turn_list': 2,
            # 'winner_game_ids': [g.decode('ascii') for g in inputs['winner_guids'][start:end]],#.squeeze(0),
            'winner_fen_arrays': fen_len,
            'winner_is_en_passant_list': 2,
            'winner_is_castling_list': 3,
            'winner_perspective_token': 2,
            
            'loser_move_froms': from_len,
            'loser_move_tos': to_len,
            # 'loser_pieces': inputs['loser_pieces'],
            'loser_winners': 2,
            'loser_turn_list': 2,
            # 'loser_game_ids': [g.decode('ascii') for g in inputs['loser_guids'][start:end]],#.squeeze(0),
            'loser_fen_arrays': fen_len,
            'loser_is_en_passant_list': 2,
            'loser_is_castling_list': 3,
            'loser_perspective_token':2
        }
pad_tokens = {
    'winner_move_froms': from_len -3,     # Invalid square (0-63 are valid)
    'winner_move_tos': to_len-3,       # Invalid square  
    'loser_move_froms': from_len -3,     # Invalid square (0-63 are valid)
    'loser_move_tos': to_len-3,       # Invalid square  
    'white_move_froms': from_len -3,     # Invalid square (0-63 are valid)
    'white_move_tos': to_len-3,       # Invalid square  
    'black_move_froms': from_len -3,     # Invalid square (0-63 are valid)
    'black_move_tos': to_len-3,       # Invalid square  
}



def create_multi_token_attention_mask(inputs, pad_tokens):
    """
    Create attention mask from multiple token types with different padding values
    
    Args:
        inputs: Dict of input tensors, e.g., {'moves': tensor, 'move_froms': tensor, 'move_tos': tensor}
        pad_tokens: Dict mapping input keys to their padding token values
                   e.g., {'moves': 0, 'move_froms': 64, 'move_tos': 64}
    
    Returns:
        attention_mask: (B, T) tensor where 1=valid, 0=should be masked
    """
    # Start with None, will be set by first valid input
    attention_mask = None
    
    for key, tensor in inputs.items():
        if key in pad_tokens:
            pad_token = pad_tokens[key]
            # Create mask for this token type: 1 where NOT padding, 0 where padding
            token_mask = (tensor < pad_token).float()  # (B, T)
            
            if attention_mask is None:
                attention_mask = token_mask
            else:
                # Combine masks: position is valid only if ALL token types are valid
                attention_mask = attention_mask * token_mask
    
    return attention_mask




def collate_indexed_games(batch, vocab_sizes=vocab_sizes, pad_token_ids = None, pad_token_thresholds=pad_tokens):
    """
    Collate function with proper padding handling for inputs and targets
    
    Args:
        batch: List of (inputs, targets) tuples
        vocab_sizes: Dict mapping key names to vocabulary sizes (to determine padding tokens)
        pad_token_ids: Dict mapping key names to their specific padding token values
    """
    inputs_list, targets_list = zip(*batch)
    keys = list(inputs_list[0].keys())
    tar_keys = list(targets_list[0].keys())
    
    collated_inputs = {}
    collated_targets = {}
    
    # Get sequence lengths
    first_key = keys[0]
    input_lengths = torch.tensor([len(item[first_key]) for item in inputs_list])
    max_len = input_lengths.max().item()
    
    # Process inputs
    for key in keys:
        input_seqs = [item[key] for item in inputs_list]
        
        # Determine padding value for this key
        if pad_token_ids and key in pad_token_ids:
            # Use explicitly provided padding token
            input_padding_value = pad_token_ids[key]
        elif vocab_sizes and key in vocab_sizes:
            # Use vocab_size as padding token (one beyond valid range)
            input_padding_value = vocab_sizes[key]
        else:
            # Fallback: find max value in this batch and use max+1
            if len(input_seqs) > 0 and all(len(seq) > 0 for seq in input_seqs):
                max_val_in_batch = max(seq.max().item() for seq in input_seqs)
                input_padding_value = max_val_in_batch + 1
                print(f"Warning: Using {input_padding_value} as padding for {key}. "
                      f"Consider providing vocab_sizes or pad_token_ids.")
            # else:
            #     input_padding_value = 0  # Safe fallback
        
        # Pad sequences
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_seqs, 
            batch_first=True, 
            padding_value=input_padding_value
        )
        
        collated_inputs[key] = padded_inputs
    
    # Process targets
    for key in tar_keys:
        target_seqs = [item[key] for item in targets_list]
        
        # Targets always use -100 (ignored by CrossEntropyLoss)
        target_padding_value = -100
        
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            target_seqs, 
            batch_first=True, 
            padding_value=target_padding_value
        )
        
        collated_targets[key] = padded_targets
    
    # Create attention mask combining sequence length and token-specific padding
    # 1. Basic sequence length mask
    length_mask = torch.arange(max_len).expand(len(input_lengths), -1) < input_lengths.unsqueeze(1)
    
    # 2. Token-specific padding mask
    if pad_token_ids:
        token_pad_mask = create_multi_token_attention_mask(collated_inputs, pad_token_thresholds)
        # Combine both masks: valid if both length and token conditions are met
        attention_mask = length_mask.float() * token_pad_mask
    else:
        attention_mask = length_mask.float()
    
    return collated_inputs, collated_targets, attention_mask



class MultiHeadAttention4DVectorized(nn.Module):
    def __init__(self, n_heads, head_size,fen_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.fen_len = fen_vocab_size
        
        # Projection for Q, K, V - process all T at once
        self.proj = nn.Linear(head_size, head_size * 3)
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(head_size, head_size)
        
        # Query vector for aggregation (to reduce L dimension to 1)
        self.agg_query = nn.Parameter(torch.randn(1, 1, fen_vocab_size))

    def forward(self, x):
        """
        Input: (B, T, L, C)
        Output: (B, T, C)
        """
        B, T, L, C = x.shape
        
        # Reshape to process all T at once: (B, T, L, C) -> (B * T, L, C)
        x_reshaped = x.reshape(B * T,L, C)
        
        # Project to Q, K, V
        Q,K, V = self.proj(x_reshaped).chunk(3, dim=-1)  # Each (B*T, L, head_size)
        
        # # Use learned query for aggregation
        # agg_q = self.agg_query.expand(B * T, 1, self.head_size)  # (B*T, 1, head_size)
        
        # Attention scores
        attn_scores = Q @ K.transpose(-2, -1) * self.head_size**-0.5  # B*T,L,C @ B*T,C,L > B*T, L, L
        
        
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # B*T, L, L
        # Weighted sum
        aggregated = attn_weights @ V  # (B*T, L, C)
        agg_q = self.agg_query.expand(B * T, 1, self.fen_len)  # (B*T, 1, L)
        aggregated = agg_q@ aggregated ## (B*T, 1, head_size)

        # Final projection
        output = self.dropout(self.output_proj(aggregated.view(B,T,C)))  # (B*T, 1, head_size)
        
        # Reshape back: (B*T, 1, head_size) -> (B, T, head_size)
        final_output = output.view(B, T, self.head_size)
        
        return final_output

class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads, head_size):
        self.n_heads = n_heads
        self.head_size = head_size
        super().__init__()
        # Projections for blocks (keys/values) and queries (letters)
        self.block_proj = nn.Linear(n_embd, n_embd * 3)  # Q, K, V

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, blocks, attention_mask =None):
        """
        Inputs:
            blocks:  (batch, block_len, n_embd)  # All blocks concatenated
            queries: (batch, query_len, n_embd)  # Query letters
        """
        # batch_size = blocks.size(0)
        B,T,C = blocks.shape
        
        # 1. Project blocks to K/V and queries to Q
        Q, K, V = self.block_proj(blocks).chunk(3, dim=-1)  # Each (batch, block_len, n_embd)

        
        # 2. Split into heads
        Q = Q.view(B, -1, self.n_heads, self.head_size).transpose(1, 2)  
        K = K.view(B, -1, self.n_heads, self.head_size).transpose(1, 2) 
        V = V.view(B, -1, self.n_heads, self.head_size).transpose(1, 2)  
        # 3. Attention scores between queries and blocks
        attn_scores = Q @ K.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T) 
        # attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Start with causal mask
        causal_mask = self.tril[:T, :T]  # (T, T)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask shape: (B, T) where 1 = valid, 0 = padding
            
            # Method 1: Create combined mask
            # First expand causal mask to batch dimension
            combined_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)
            
            # Create padding mask for attention
            # We need to mask positions where either the query OR key is padding
            padding_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # (B, T, T)
            
            # Combine: position is valid if both causal and padding allow it
            combined_mask = combined_mask * padding_mask  # (B, T, T)
            
            # Expand for multi-head
            combined_mask = combined_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # (B, nh, T, T)
            
            # Apply mask
            attn_scores = attn_scores.masked_fill(combined_mask == 0, float('-inf'))
        else:
            # Only causal mask
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        
        # 5. Softmax + Head Mixing
        attn_weights = F.softmax(attn_scores, dim=-1)
        # 6. Weighted sum of block values
        output = attn_weights @ V
        output = output.transpose(1, 2).reshape(B, -1, self.n_heads * self.head_size)

        output = self.dropout(self.proj(output))
        
        return output  # (batch, query_len, n_embd)

class CrossMultiHeadAttention(nn.Module):
    def __init__(self,n_heads, head_size,):
        self.n_heads = n_heads
        self.head_size = head_size
        super().__init__()
        # Projections for blocks (keys/values) and queries (letters)
        self.query_proj = nn.Linear(n_embd, n_embd)  # Q, K, V
        self.key_proj =  nn.Linear(n_embd, n_embd)
        self.value_proj =  nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, query, key, value, attention_mask =None):
        """
        Inputs:
            blocks:  (batch, block_len, n_embd)  # All blocks concatenated
            queries: (batch, query_len, n_embd)  # Query letters
        """
        # batch_size = blocks.size(0)
        B,T,C = query.shape
        
        # 1. Project blocks to K/V and queries to Q
        Q, K, V = self.query_proj(query), self.key_proj(key),self.value_proj(value)  # Each (batch, block_len, n_embd)

        
        # 2. Split into heads
        Q = Q.view(B, -1, self.n_heads, self.head_size).transpose(1, 2)  
        K = K.view(B, -1, self.n_heads, self.head_size).transpose(1, 2) 
        V = V.view(B, -1, self.n_heads, self.head_size).transpose(1, 2)  
        # 3. Attention scores between queries and blocks
        attn_scores = Q @ K.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T) 
        # attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    
        # Start with causal mask
        causal_mask = self.tril[:T, :T]  # (T, T)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask shape: (B, T) where 1 = valid, 0 = padding
            
            # Method 1: Create combined mask
            # First expand causal mask to batch dimension
            combined_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)
            
            # Create padding mask for attention
            # We need to mask positions where either the query OR key is padding
            padding_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # (B, T, T)
            
            # Combine: position is valid if both causal and padding allow it
            combined_mask = combined_mask * padding_mask  # (B, T, T)
            
            # Expand for multi-head
            combined_mask = combined_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # (B, nh, T, T)
            
            # Apply mask
            attn_scores = attn_scores.masked_fill(combined_mask == 0, float('-inf'))
        else:
            # Only causal mask
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 5. Softmax + Head Mixing
        attn_weights = F.softmax(attn_scores, dim=-1)
        # 6. Weighted sum of block values
        output = attn_weights @ V
        output = output.transpose(1, 2).reshape(B, -1, self.n_heads * self.head_size)

        output = self.dropout(self.proj(output))
        
        return output  # (batch, query_len, n_embd)



class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x,attention_mask=None):
        x = x + self.sa(self.ln1(x), attention_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class MoveCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Project both move components to same space
        self.from_proj = MultiHeadAttention(n_head, head_size)
        self.to_proj = MultiHeadAttention(n_head, head_size)
        
        # Cross-attention: from attends to to, and vice versa
        self.cross_attn = CrossMultiHeadAttention(n_head, head_size)
    
    def forward(self, move_from_emb, move_to_emb):
        from_emb = self.from_proj(move_from_emb,attention_mask)
        to_emb = self.to_proj(move_to_emb,attention_mask)
        
        # Cross-attention: from positions attend to to positions
        from_enhanced = self.cross_attn(from_emb, to_emb, to_emb,attention_mask)
        to_enhanced = self.cross_attn(to_emb, from_emb, from_emb,attention_mask)
        
        return from_enhanced + to_enhanced

batch_size = 2#16#*4
block_size = 256 #32#*4
n_embd = 64#*2
dropout=0.1
fen_len =pieces_num
perspective_vocab_size = 2
pieces_num = fen_len = len(pieces_dict)+1
fen_vocab_size = 64

n_head = 4
n_layer = 4
head_size = n_embd // n_head
num_players = 2
epochs = 2#000
eval_interval = 100
learning_rate = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np



"""
Test function for the indexed chess dataset
"""


def sample_from_dataloader(dataloader, num_samples):
    """
    Sample random batches from a dataloader (works with shuffle=True)
    """
    losses = torch.zeros(num_samples)
    dataloader_iter = iter(dataloader)
    
    for i in range(num_samples):
        try:
            batch, target,attention_mask = next(dataloader_iter)
        except StopIteration:
            # Reset iterator if we reach the end
            dataloader_iter = iter(dataloader)
            batch, target,attention_mask = next(dataloader_iter)
            # Move batch and target to GPU
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        target = {k: v.to(device) if torch.is_tensor(v) else v 
                 for k, v in target.items()}
        
        # Your model forward pass
        white_from_logits, white_to_logits, black_from_logits, black_to_logits, loss = model(batch, target,attention_mask.to(device))

        losses[i] = loss.item()
    
    return losses


# Alternative version if your test_dataset returns batches instead of individual samples
def estimate_loss(epoch):
    """
    Alternative version if test_dataset returns batches directly
    """
    out = {}
    model.eval()
    
    
    
    with torch.no_grad():
        for split in ['train', 'val']:
            if split == 'train':
                losses = sample_from_dataloader(train_dataloader, eval_iters)
            else:
                # For test data, we need to handle batches differently
                losses = torch.zeros(eval_iters)
                for i in range(eval_iters):
                    # Get batch_size random indices
                    rng = torch.Generator().manual_seed(42 + epoch*eval_iters + i)
                    indices = torch.randint(0, len(test_dataset), (batch_size,), generator=rng)
                    
                    # Collect batch samples
                    batch_samples = []
                    target_samples = []
                    
                    for idx in indices:
                        batch_sample, target_sample = test_dataset[idx]
                        batch_samples.append(batch_sample)
                        target_samples.append(target_sample)
                    
                    # # Stack individual samples into batches
                    # batch = {
                    #     'winner_perspective_token': torch.stack([b['winner_perspective_token'] for b in batch_samples]),
                    #     'winner_fen_arrays': torch.stack([b['winner_fen_arrays'] for b in batch_samples]),
                    #     'winner_winners': torch.stack([b['winner_winners'] for b in batch_samples]),
                    #     'winner_move_froms': torch.stack([b['winner_move_froms'] for b in batch_samples]),
                    #     'winner_move_tos': torch.stack([b['winner_move_tos'] for b in batch_samples]),
                    #     'winner_is_castling_list': torch.stack([b['winner_is_castling_list'] for b in batch_samples]),
                    #     'winner_is_en_passant_list': torch.stack([b['winner_is_en_passant_list'] for b in batch_samples]),


                    #     'loser_perspective_token': torch.stack([b['loser_perspective_token'] for b in batch_samples]),
                    #     'loser_fen_arrays': torch.stack([b['loser_fen_arrays'] for b in batch_samples]),
                    #     'loser_winners': torch.stack([b['loser_winners'] for b in batch_samples]),
                    #     'loser_move_froms': torch.stack([b['loser_move_froms'] for b in batch_samples]),
                    #     'loser_move_tos': torch.stack([b['loser_move_froms'] for b in batch_samples]),
                    #     'loser_is_castling_list': torch.stack([b['loser_is_castling_list'] for b in batch_samples]),
                    #     'loser_is_en_passant_list': torch.stack([b['loser_is_en_passant_list'] for b in batch_samples]),

                    # }
                    
                    # target = {
                    #     'white_move_tos': torch.stack([t['white_move_tos'] for t in target_samples]),
                    #     'white_move_froms': torch.stack([t['white_move_froms'] for t in target_samples]),
                    #     'black_move_tos': torch.stack([t['black_move_tos'] for t in target_samples]),
                    #     'black_move_froms': torch.stack([t['black_move_froms'] for t in target_samples])
                    # }
                    batch_inputs = [b for b in batch_samples]  # List of input dicts
                    batch_targets = [t for t in target_samples]  # List of target dicts
                    
                    # Create the batch tuples format expected by collate function
                    batch_tuples = list(zip(batch_inputs, batch_targets))
                    
                    # Use your collate function to handle padding
                    batch, target, attention_mask = collate_indexed_games(
                        batch_tuples, 
                        # vocab_sizes=vocab_sizes, 
                        # pad_token_thresholds=pad_tokens
                    )
                    # Move batch and target to GPU
                    batch = {k: v.to(device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    target = {k: v.to(device) if torch.is_tensor(v) else v 
                             for k, v in target.items()}
                    # test_batch_tuple = (batch,target)
                    # batch, target, attention_mask = collate_indexed_games(test_batch_tuple)
                    
                    white_from_logits, white_to_logits, black_from_logits, black_to_logits, loss = model(batch, target, attention_mask.to(device))
                    

                    losses[i] = loss.item()
            
            out[split] = losses.mean()
    
    model.train()
    return out

class ChessTransformerCoder(nn.Module):
    def __init__(self):#, move_vocab_size, perspective_vocab_size, fen_len, n_embd, n_head, n_layer):
        super().__init__()        
        # # Separate Embedding Layers for different input types
        self.move_from_embedder = nn.Embedding(from_len+1, n_embd) # For past moves (if you use them)
        self.move_to_embedder = nn.Embedding(to_len+1, n_embd) # For past moves (if you use them)
        self.is_castling_embedder = nn.Embedding(3+1, n_embd) # For past moves (if you use them)
        self.is_en_passant_embedder = nn.Embedding(2+1, n_embd) # For past moves (if you use them)

        self.perspective_embedder = nn.Embedding(perspective_vocab_size+1, n_embd) # For [white_winning], etc.
        self.pieces_embedder = nn.Embedding(pieces_num+1, n_embd)
        self.winner_embedder = nn.Embedding(num_players+1, n_embd)

        self.board_proj = MultiHeadAttention4DVectorized(n_heads= block_size, head_size=n_embd,fen_len=fen_vocab_size) # For the FEN array
        
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.fen_position_embedding_table = nn.Embedding(fen_vocab_size, n_embd)
        # self.piece_attn = MultiHeadAttention(n_head, head_size)
        # Learn how to weight each embedding type
        self.perspective_weight = nn.Parameter(torch.tensor(0.2))
        self.board_persp_ln = nn.LayerNorm(n_embd)
        self.move_from_ln = nn.LayerNorm(n_embd)
        self.move_to_ln = nn.LayerNorm(n_embd)
        self.move_ln = nn.LayerNorm(n_embd)
        self.board_weight = nn.Parameter(torch.tensor(0.2))
        self.pieces_weight = nn.Parameter(torch.tensor(0.2))
        self.winner_weight = nn.Parameter(torch.tensor(0.2))
        self.move_weight = nn.Parameter(torch.tensor(0.2))


        self.board_attn = MultiHeadAttention(n_head, head_size)
        self.move_attn = MultiHeadAttention(n_head, head_size)
        self.move_from_attn = MultiHeadAttention(n_head, head_size)
        self.move_to_attn = MultiHeadAttention(n_head, head_size)
        self.move_context_ln = nn.LayerNorm(n_embd)
        self.move_context_cross_attention = CrossMultiHeadAttention(n_head, head_size)
        # self.attention_layer1 = MultiHeadAttention(n_head, head_size)
        # self.attention_layer2 = MultiHeadAttention(n_head, head_size)
        # self.cross_attention_layer = CrossMultiHeadAttention(n_head, head_size)
        # self.move_cross_attn = MoveCrossAttention()
        

        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        #nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # self.lm_head0 = nn.Linear(n_embd, from_len)
        # self.rms_norm1 = nn.RMSNorm(n_embd)
        # self.rms_norm2 = nn.RMSNorm(n_embd)
        # self.lm1_proj = nn.Linear(2*n_embd, n_embd)
        # self.lm1_block = Block(n_embd, n_head=n_head)

        
        # # The head to predict the next MOVE
        # self.lm_head1 = nn.Linear(n_embd, to_len)
    def _calculate_loss(self, logits, targets, batch_size, time_steps):
        if targets is None:
            return 0
        targets_flat = targets.view(batch_size * time_steps)
        return F.cross_entropy(logits, targets_flat,label_smoothing=0.2)



    def forward(self, perspective_ids, fen_arrays,winners, move_froms, move_tos, is_castling, is_en_passant, attention_mask = None):
    # Check for NaN in inputs
        for tensor in [perspective_ids, fen_arrays, winners, move_froms, move_tos]:
            if torch.isnan(tensor).any():
                print("NaN in input!")
                return None, None, torch.tensor(float('nan'))
        # print(attention_mask)

        # Add time dimension if missing (single sample case)
        if len(perspective_ids.shape) == 1:
            perspective_ids = perspective_ids.unsqueeze(0)  # [B] -> [1, B]
            fen_arrays = fen_arrays.unsqueeze(0)  # [B, F] -> [1, B, F]
            if targets_move_tos is not None:
                targets_move_tos = targets_move_tos.unsqueeze(0)
            if targets_move_froms is not None:
                targets_move_froms = targets_move_froms.unsqueeze(0)
        B, T = perspective_ids.shape
        # print(B)
        # print(T)
        # print(fen_arrays.shape)
        
        # Embed each input modality separately
        perspective_emb = self.perspective_embedder(perspective_ids)#* self.perspective_weight # shape: B,T,C
        fen_positions = self.fen_position_embedding_table (torch.arange(fen_vocab_size, device=device)).unsqueeze(0).unsqueeze(0)
        # print(fen_vocab_size)
        # print(fen_positions.shape)
        board_emb = self.board_proj(self.pieces_embedder(fen_arrays)+ fen_positions)#* self.board_weight # shape: B,T,C
        # print(board_emb.shape)
        winner_emb = self.winner_embedder(winners) #* self.winner_weight
        # pieces_emb = self.pieces_embedder(pieces)
        move_from_em = self.move_from_embedder(move_froms)
        move_from_emb = self.move_from_ln(self.move_from_attn(move_from_em,attention_mask) + move_from_em)
        move_to_em = self.move_to_embedder(move_tos)
        move_to_emb = self.move_to_ln(self.move_to_attn(move_to_em,attention_mask) + move_to_em)
        is_en_passant_emb = self.is_en_passant_embedder(is_en_passant)
        is_castling_emb = self.is_castling_embedder(is_castling)
        move_emb = move_from_emb + move_to_emb + is_en_passant_emb + is_castling_emb
        move_attn = self.move_ln(self.move_attn(move_emb,attention_mask) +move_emb)#* self.move_weight
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        # pieces_attn = self.piece_attn(pieces_emb) #* self.pieces_weight
        board_persp_emb = perspective_emb + board_emb + pos_emb + winner_emb # shape: [batch_size, block_size, n_embd]
        board_persp_attn = self.board_persp_ln(self.board_attn(board_persp_emb,attention_mask) + board_persp_emb)
        move_context = self.move_context_ln(self.move_context_cross_attention(query=move_attn, key=board_persp_attn, value= board_persp_attn,attention_mask = attention_mask) + move_attn)

        #add position emb, perspective emb and board emb together.         
        input_emb = board_persp_attn + move_context# + pieces_attn
        # x = self.blocks(input_emb,attention_mask) #+ input_emb # (B,T,C)
        
        if isinstance(self.blocks, nn.Sequential):
            x = self.blocks(input_emb)  # Only pass input, not attention_mask
        
        
        elif isinstance(self.blocks, nn.ModuleList):
            for block in self.blocks:
                x = block(input_emb, attention_mask=attention_mask)
        x = self.ln_f(x) # (B,T,C)

        return x





class ChessTransformer(nn.Module):
    def __init__(self):#, move_vocab_size, perspective_vocab_size, fen_len, n_embd, n_head, n_layer):
        super().__init__()        
        # # Separate Embedding Layers for different input types
        self.loser_encoder = ChessTransformerCoder()
        self.winner_encoder = ChessTransformerCoder()

        self.winner_cross_attention = CrossMultiHeadAttention(n_head, head_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln2 = nn.LayerNorm(n_embd)
        self.lm_head0 = nn.Linear(n_embd, from_len-3)
        self.rms_norm1 = nn.RMSNorm(n_embd)
        self.rms_norm2 = nn.RMSNorm(n_embd)
        self.lm1_proj = nn.Linear(n_embd, n_embd)
        self.lm1_block = Block(n_embd, n_head=n_head)
        # The head to predict the next MOVE
        self.lm_head1 = nn.Linear(n_embd, to_len-3)
        self.rms_norm3 = nn.RMSNorm(n_embd)
        # self.rms_norm4 = nn.RMSNorm(n_embd)
        self.lm2_proj = nn.Linear(n_embd, n_embd)
        self.lm2_block = Block(n_embd, n_head=n_head)
        # The head to predict the next MOVE
        self.lm_head2 = nn.Linear(n_embd, from_len-3)
        # self.rms_norm5 = nn.RMSNorm(n_embd)
        # self.rms_norm6 = nn.RMSNorm(n_embd)
        self.lm3_proj = nn.Linear(n_embd, n_embd)
        self.lm3_block = Block(n_embd, n_head=n_head)
        # The head to predict the next MOVE
        self.lm_head3 = nn.Linear(n_embd, to_len-3)
    def _calculate_loss(self, logits, targets, batch_size, time_steps):
        if targets is None:
            return 0

        vocab_size = logits.shape[-1]
        
        targets_flat = targets.view(batch_size * time_steps)
        return F.cross_entropy(logits, targets_flat,label_smoothing=0.1, ignore_index=-100)



    def forward(self, batch, targets=None, attention_mask=None):

        
        winner_attn = self.loser_encoder(
            perspective_ids=batch['winner_perspective_token'],
            fen_arrays=batch['winner_fen_arrays'],
            winners=batch['winner_winners'],
            move_froms = batch['winner_move_froms'],
            move_tos= batch['winner_move_tos'],
            is_castling = batch['winner_is_castling_list'],
            is_en_passant = batch['winner_is_en_passant_list'],
            attention_mask =attention_mask
            # targets_move_tos=target['winner_move_tos'],
            # targets_move_froms=target['winner_move_froms']
        )
        
        loser_attn = self.winner_encoder(
            perspective_ids=batch['loser_perspective_token'],
            fen_arrays=batch['loser_fen_arrays'],
            winners=batch['loser_winners'],
            move_froms = batch['loser_move_froms'],
            move_tos = batch['loser_move_tos'],
            is_castling = batch['loser_is_castling_list'],
            is_en_passant = batch['loser_is_en_passant_list'],
            attention_mask =attention_mask
            # targets_move_tos=target['loser_move_tos'],
            # targets_move_froms=target['loser_move_froms']
        )

        winner_loser_cross_attn = self.ln1(self.winner_cross_attention(query= winner_attn, key=loser_attn, value=loser_attn) + winner_attn)
        # x = self.blocks(winner_loser_cross_attn,attention_mask)  
        # If blocks is Sequential (can't handle attention_mask)
        if isinstance(self.blocks, nn.Sequential):
            x = self.blocks(winner_loser_cross_attn)  # Only pass input, not attention_mask
        
        # If blocks is ModuleList (can handle attention_mask)
        elif isinstance(self.blocks, nn.ModuleList):
            for block in self.blocks:
                x = block(winner_loser_cross_attn, attention_mask=attention_mask)

        logits0 = self.lm_head0(x) # (B,T,vocab_size)
        x = self.rms_norm1(x)
        x = self.lm1_block(self.lm1_proj(x),attention_mask)                
        logits1 = self.lm_head1(x) # (B,T,vocab_size)
        
        x = self.rms_norm2(x)
        x = self.lm2_block(self.lm2_proj(x),attention_mask)                
        logits2 = self.lm_head2(x) # (B,T,vocab_size)
        
        x = self.rms_norm3(x)
        x = self.lm3_block(self.lm3_proj(x),attention_mask)                
        logits3 = self.lm_head3(x) # (B,T,vocab_size)


        B, T, C0 = logits0.shape
        _ , _, C1 = logits1.shape
        logits0 = logits0.view(B*T, C0)
        logits1 = logits1.view(B*T, C1)
        logits2 = logits2.view(B*T, C0)
        logits3 = logits3.view(B*T, C1)
        loss0 = self._calculate_loss(logits0, targets['white_move_froms'], B, T)
        loss1 = self._calculate_loss(logits1, targets['white_move_tos'], B, T)
        loss2 = self._calculate_loss(logits2, targets['black_move_froms'], B, T)
        loss3 = self._calculate_loss(logits3, targets['black_move_tos'], B, T)
        
        if targets is None:
            loss = None
        else:
            loss = loss0 + loss1 + loss2 + loss3
        
        return logits0, logits1, logits2, logits3, loss
        



from torch.optim.lr_scheduler import LambdaLR,ReduceLROnPlateau,CosineAnnealingLR
# scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=50)
# scheduler = scheduler = CosineAnnealingLR(optimizer, T_max=100)


train_dataloader_iter = cycle(train_dataloader)



for epoch in tqdm(range(epochs)):
    # if epoch == 2000:
    #     scheduler = LambdaLR(optimizer, lr_lambda)
    try:
        batch,target,attention_mask = next(train_dataloader_iter)
        # Move batch and target to GPU
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        target = {k: v.to(device) if torch.is_tensor(v) else v 
                 for k, v in target.items()}
    except StopIteration:
        train_dataloader_iter = cycle(train_dataloader)
        batch,target,attention_mask = next(train_dataloader_iter)
        # Move batch and target to GPU
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        target = {k: v.to(device) if torch.is_tensor(v) else v 
                 for k, v in target.items()}

    # every once in a while evaluate the loss on train and val sets
    if epoch % eval_interval == 0 or epoch == epochs - 1:
        
        losses = estimate_loss(epoch)
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    optimizer.zero_grad(set_to_none=True)

    white_from_logits, white_to_logits, black_from_logits, black_to_logits, loss = model(batch, target,attention_mask.to(device))

    if torch.isnan(loss).any():
        print(f"NaN detected in loss!{epoch}")
        # Check individual components
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN in {name}")
    
    
    # Combine the losses
    loss.backward()

    optimizer.step()


    # scheduler.step()

