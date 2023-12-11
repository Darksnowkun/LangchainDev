import os
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def my_example():
    # This is the repo location of the model being used
    model="Helsinki/opus-mt-en-zh"
    model="Helsinki-NLP/opus-mt-zh-en"

    # This is the sentence to be translated
    question = "Hello my name is Elder John and I would like to share with you this most amazing book"
    question = "你好我叫"

    # Setup from repos readme

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

    # Tokenize the input
    print("---")
    print("Inputs: ")
    print(inputs)
    print()
    gen_tokens = model.generate(**inputs, use_cache=True)
    print("Gen Tokens: ")
    print(gen_tokens)
    print()

    outputs = tokenizer.batch_decode(gen_tokens)
    print("Output:")
    print(outputs)

# my_example()

def my_example_2():
    model="google/flan-t5-base"
    question="Give me an accurate statement about the year 1955."
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)
    tokenized_input = tokenizer(question, return_tensors="pt")
    tokenized_output = model.generate(**tokenized_input, use_cache=True)
    decoded_output = tokenizer.batch_decode(tokenized_output)
    print(decoded_output)

my_example_2()

#--------------------#

def working_example():
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

    article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
    article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."
    my_text = "Hello my name is Elder John and I would like to share with you this most amazing book"

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    # translate Hindi to French
    tokenizer.src_lang = "hi_IN"
    encoded_hi = tokenizer(article_hi, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
    )
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    # => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

    # translate Arabic to English
    tokenizer.src_lang = "ar_AR"
    encoded_ar = tokenizer(article_ar, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_ar,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

    # translate English to Chinese
    tokenizer.src_lang = "en_XX"
    encoded_text = tokenizer(my_text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
    )
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

    # translate English to Vietnamese
    tokenizer.src_lang = "en_XX"
    encoded_text = tokenizer(my_text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"]
    )
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

    # => "The Secretary-General of the United Nations says there is no military solution in Syria."
