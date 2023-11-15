import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor, AutoModel
import streamlit as st
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
st.title("Сокращение текста и прослушивание результата")
article_text=st.text_input('Введите текст и нажмите Enter')
#article_text = """Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said.  The policy includes the termination of accounts of anti-vaccine influencers.  Tech giants have been criticised for not doing more to counter false health information on their sites.  In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue.  YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines.  In a blog post, the company said it had seen false claims about Covid jabs "spill over into misinformation about vaccines in general". The new policy covers long-approved vaccines, such as those against measles or hepatitis B.  "We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO," the post said, referring to the World Health Organization."""

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_ids = tokenizer(
    [WHITESPACE_HANDLER(article_text)],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=84,
    no_repeat_ngram_size=2,
    num_beams=4
)[0]

summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
st.text(summary)
st.title("Аудио")
#print(summary)

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

inputs = processor(
    text=summary,
    return_tensors="pt",
)

speech_values = model.generate(**inputs, do_sample=True)

sampling_rate = model.generation_config.sample_rate

if speech_values is not None and len(speech_values) > 0:
    sampling_rate = model.generation_config.sample_rate
    st.audio(speech_values.cpu().numpy().squeeze(), format="audio/wav", start_time=0, sample_rate=sampling_rate)
else:
    st.warning("No audio generated.")

#st.audio(Audio(speech_values.cpu().numpy().squeeze(), rate=sampling_rate))
#Audio(speech_values.cpu().numpy().squeeze(), rate=sampling_rate)
