# Voice2Text
Transcribe audio files to text

## Usage
1. Initialize 3 folders: `output_texts`, `source_data`, `temp`
2. Prepare the Claude API key and Huggingface API key
3. Place your audio file in the `source_data` folder. The audio format should be .m4a. If you want to use another format, please edit the code. (Currently, only one file at a time.)
4. The result will be in the `output_texts` folder.
5. The audio duration should ideally be less than 60 minutes.

## Cost
1. The `openai/whisper_large_v3` model requires a GPU with more than 16GB of memory.
2. Transcribing a 1-hour audio file might cost between $0.10 and $0.20 using Claude 3 Sonnet.

## Notice
1. The pre-processing step only amplifies the speaker's voice and normalizes the audio; no noise reduction is performed.
2. The process can be improved if desired.
