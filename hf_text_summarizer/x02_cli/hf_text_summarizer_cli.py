from transformers import BartForConditionalGeneration, BartTokenizer

# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], 
                                    num_beams=4,
                                    min_length=20, 
                                    #max_length=40, 
                                    early_stopping=True)

    # Convert summary IDs to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

text = """Paris is the capital and most populous city of France, with
          an estimated population of 2,175,601 residents as of 2018,
          in an area of more than 105 square kilometres (41 square
          miles). The City of Paris is the centre and seat of
          government of the region and province of Île-de-France, or
          Paris Region, which has an estimated population of
          12,174,880, or about 18 percent of the population of France
          as of 2017."""

summary = summarize_text(text)
print(summary)