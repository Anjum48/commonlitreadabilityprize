import sentencepiece as spm
import src.sentencepiece_pb2 as pb2


class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        sp_model = str(next(model_path.glob("**/*.model")))
        print("Loaded sentencepiece model", sp_model)
        self.sp.load(sp_model)

    def encode(self, text):
        spt = pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(text))
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
        return [2] + tokens + [3]

    # Match the call signature for AutoTokenizer.encode_plus
    def encode_plus(
        self,
        text,
        return_token_type_ids=True,
        return_offsets_mapping=True,
        add_special_tokens=True,  # Not used
        model_max_length=-1,  # TODO: Not used
        pad_to_max_length=False,  # TODO: Not used
    ):

        spt = pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(text))
        offsets = []
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
            offsets.append((piece.begin, piece.end))

        if model_max_length != -1:
            tokens = tokens[:model_max_length]
            offsets = offsets[:model_max_length]

        output = {
            "input_ids": [2] + tokens + [3],
            "attention_mask": [1] * (len(tokens) + 2),
        }

        if return_token_type_ids:
            output["token_type_ids"] = [0] * (len(tokens) + 2)

        if return_offsets_mapping:
            output["offset_mapping"] = [None] + offsets + [None]

        return output
