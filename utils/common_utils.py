from string import punctuation
import os


class Logging:
    def __init__(self, file_name):
        self.file_name = file_name
        # 确保日志文件的目录存在
        log_dir = os.path.dirname(file_name)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)  # 创建目录
        # 如果文件不存在，创建文件
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                f.write("")  # 创建空文件

    def logging(self, info):
        with open(self.file_name, 'a+') as f:
            print(info, file=f)
        print(info)



def get_stop_words():
    stop_words = []#set()
    for i in punctuation:
        stop_words.append('Ġ' + i)
    return stop_words

stop_words = get_stop_words()

