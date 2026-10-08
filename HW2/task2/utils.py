import torch


class LMCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, outputs, tokens, tokens_lens, loss_mask=None):
        """
        :param torch.Tensor outputs: Output from LM.forward. Shape: [B, T, V]
        :param torch.Tensor tokens: Batch of tokens. Shape: [B, T]
        :param torch.Tensor tokens_lens: Length of each sequence in batch
        :param torch.Tensor loss_mask: Valid next-token transitions. Shape: [B, T - 1]
        :return torch.Tensor: CrossEntropyLoss between corresponding logits and tokens
        """
        logits = outputs[:, :-1]
        targets = tokens[:, 1:]

        if loss_mask is not None:
            mask = loss_mask.to(device=tokens.device, dtype=torch.bool)
        else:
            tokens_lens = torch.as_tensor(tokens_lens, device=tokens.device)
            positions = torch.arange(tokens.shape[1] - 1, device=tokens.device)
            mask = positions[None, :] < (tokens_lens[:, None] - 1)

        return super().forward(logits[mask], targets[mask])


class LMAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, tokens, tokens_lens, loss_mask=None):
        """
        :param torch.Tensor outputs: Output from LM.forward. Shape: [B, T, V]
        :param torch.Tensor tokens: Batch of tokens. Shape: [B, T]
        :param torch.Tensor tokens_lens: Length of each sequence in batch
        :param torch.Tensor loss_mask: Valid next-token transitions. Shape: [B, T - 1]
        :return torch.Tensor: Accuracy for given logits and tokens
        """
        predictions = outputs[:, :-1].argmax(dim=-1)
        targets = tokens[:, 1:]

        if loss_mask is not None:
            mask = loss_mask.to(device=tokens.device, dtype=torch.bool)
        else:
            tokens_lens = torch.as_tensor(tokens_lens, device=tokens.device)
            positions = torch.arange(tokens.shape[1] - 1, device=tokens.device)
            mask = positions[None, :] < (tokens_lens[:, None] - 1)

        correct = ((predictions == targets) & mask).sum()
        total = mask.sum().clamp(min=1)
        return correct.float() / total
