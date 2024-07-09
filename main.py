from sw_reporter import Swreporter

if __name__ == '__main__':
    editor = Swreporter(audio_path='./source_data',
                          temp_output_dir='./temp',
                          output_dir='./output_texts')
    editor.run()