from clip import clip
import torch
import torch.nn as nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy 

_tokenizer = _Tokenizer()

class SkelMaPLe(nn.Module):
    def __init__(self, input_size, output_size, 
                 dropout=.5, activation=nn.SiLU()):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        
    def forward(self, inputs):
        # analyze input
        x, compound_prompts, counter = inputs
        # preprocess
        x = self.dropout(self.norm(x))
        # maple
        if not (counter > len(compound_prompts) - 1):
            vis_context = compound_prompts[counter]
            vis_context = vis_context.expand(x.shape[0], -1)
            x = x + vis_context
            counter += 1
        else:
            x = x
        # linear
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x, compound_prompts, counter
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model, dual=False):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.dual = dual

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        # exit()
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        if self.dual:
            combined = [x, x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
            outputs = self.transformer(combined)
            x = outputs[1]  # extract the x back from here
        else:
            combined = [x, compound_prompts_deeper_text, 0]
            outputs = self.transformer(combined)
            x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
def load_clip_to_cpu(root, model_name, use_text_level_prompt=False, re_attention=False, n_ctx = 2, prompt_type=2):
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    if use_text_level_prompt:
        if model_name == 'dual_clip':
            design_details = {"trainer": 'Dual',
                            "vision_depth": 0,
                            "language_depth": 0, "vision_ctx": 0,
                            "language_ctx": 0,
                            "maple_length": n_ctx,
                            "re_attention": re_attention,
                            "prompt_type":prompt_type}
            print("Design Details:")
            print(design_details)
        else:
            design_details = {"trainer": 'MaPLe',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0,
                        "maple_length": n_ctx}
    else:
        design_details = None
    model = clip.build_model(state_dict or model.state_dict(), 
                             design_details)

    return model


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class LinearAdapter(nn.Module):
    """
    Design 2 v4
    """
    def __init__(self, inplanes, outplanes, width, act_layer=None, **kwargs):
        super().__init__()
        if act_layer is None:
            act_layer = nn.Identity
        self.fc1 = nn.Linear(inplanes, width)
        self.fc2 = nn.Linear(width, outplanes)
        self.act = act_layer()
        self.se = nn.Parameter(1.0 * torch.ones((1, outplanes)), requires_grad=True)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.norm(out)
        out = self.act(out)
        out = self.fc2(out)
        out = out * self.se
        return out