
import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, is_decoder=False):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)
        input_tensor = input_tensor.to(next(self.model.parameters()).device)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        if is_decoder:
            seq_len = input_tensor.size(1)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=input_tensor.device)).unsqueeze(0).unsqueeze(0)
        else:
            mask = None

        # Process the input tensor through the encoder model
        _,  attn_maps = self.model(input_tensor, mask=mask) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0], dim=-1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.detach().cpu().numpy())

            # Create a heatmap of the attention map
            for head_idx in range(att_map.shape[0]):
                fig, ax = plt.subplots()
                cax = ax.imshow(att_map[head_idx], cmap='hot', interpolation='nearest')
                ax.xaxis.tick_top()
                fig.colorbar(cax, ax=ax)
                plt.title(f"Attention Map {j + 1}, Head {head_idx + 1}")
            
                # Save the plot
                plt.savefig(f"attention_map_layer_{j + 1}_head_{head_idx + 1}.png")
                plt.show()
