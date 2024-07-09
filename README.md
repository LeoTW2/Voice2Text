# Voice2Text
Transcribe the audio file to texts
Usage:
  1.you need to initial 3 folder: output_texts, source_data, temp
  2.Prepare the claude api key and huggingface api key
  3.Put your audio file in source_data folder, The audio format is .m4a, if you want to use other format, please edit the code.(Currently only one file each time.)
  4.the result will be in output_texts.
  5.The audio duration should ideally be less than 60 minutes.

Cost:
  1.openai/whisper_large_v3 need to use more than 16GB GPU.
  2. 1 hour duration audio file might need 0.1~0.2 USD for Claude 3 Sonnet.
Notice:
  1.Pre-process data only louder the voice for speaker and normalize the audio, no denoise procee in.
  2.The peocess could be better if you want.
