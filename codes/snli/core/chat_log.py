import logging
import os

class SingletonType(type):
    def __call__(cls, *args, **kwargs):
        try:
            return cls.__instance
        except AttributeError:
            cls.__instance = super(SingletonType, cls).__call__(*args, **kwargs)
            return cls.__instance


class CustomLogger(object):
    __metaclass__ = SingletonType
    _logger = None

    def __init__(self):
        self._logger = logging.getLogger("cai_chatbot_framework")
        self._logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        import datetime
        now = datetime.datetime.now()
        import time
        timestamp = time.mktime(now.timetuple())

        dirname = 'data_out/logs/'
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # file_handler = logging.FileHandler(dirname + now.strftime("%Y-%m-%d %H:%M:%S")+".log")
        file_handler = logging.FileHandler(dirname + 'chatbot_framework.log')
        stream_hander = logging.StreamHandler()

        self._logger.addHandler(file_handler)
        self._logger.addHandler(stream_hander)

    def get_logger(self):
        return self._logger




# mylogger = logging.getLogger("chatbot_framwork")
# mylogger.setLevel(logging.INFO) #로깅 레벨 설정

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# #handler: 내가 로깅한 정보가 출력되는 위치 설정
# #파일 설정, default a 모드임 (a 추가)
# file_handler = logging.FileHandler('data_out/logs/chatbot_framework.log')
# stream_hander = logging.StreamHandler()

# #Handler logging 추가
# file_handler.setFormatter(formatter)
# stream_hander.setFormatter(formatter)

# #logging 추가
# mylogger.addHandler(stream_hander)
# mylogger.addHandler(file_handler)

# mylogger.info("server start!!!")



# if __name__ =='__main__':
#     # logging.info("hello world")
#     # logging.error("something wrong!")

#     mylogger = logging.getLogger("chatbot_framwork")
#     mylogger.setLevel(logging.INFO) #로깅 레벨 설정

#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     #handler: 내가 로깅한 정보가 출력되는 위치 설정
#     #파일 설정, default a 모드임 (a 추가)
#     file_handler = logging.FileHandler('data_out/logs/chatbot_framework.log')
#     stream_hander = logging.StreamHandler()

#     #Handler logging 추가
#     file_handler.setFormatter(formatter)
#     stream_hander.setFormatter(formatter)

#     #logging 추가
#     mylogger.addHandler(stream_hander)
#     mylogger.addHandler(file_handler)

#     mylogger.info("server start!!!")




# LOGGING_LEVELS = {'critical': logging.CRITICAL,
#                   'error': logging.ERROR,
#                   'warning': logging.WARNING,
#                   'info': logging.INFO,
#                   'debug': logging.DEBUG}

# def init():
#     parser = optparse.OptionParser()
#     parser.add_option('-l', '--logging-level', help='Logging level')
#     parser.add_option('-f', '--logging-file', help='Logging file name')
#     (options, args) = parser.parse_args()
#     logging_level = LOGGING_LEVELS.get(options.logging_level, logging.NOTSET)
#     logging.basicConfig(level=logging_level, filename=options.logging_file,
#                         format='%(asctime)s %(levelname)s: %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S')

# logging.debug("디버깅용 로그~~")
# logging.info("도움이 되는 정보를 남겨요~")
# logging.warning("주의해야되는곳!")
# logging.error("에러!!!")
# logging.critical("심각한 에러!!")



# def my_logger(original_function):
#     import logging
#     logging.basicConfig(filename='./logs/{}.log'.format(original_function.__name__), level=logging.INFO)
    
#     def wrapper(*args, **kwargs):
#         timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
#         logging.info(
#             '[{}] 실행결과 args - {}, kwargs - {}'.format(timestamp, args, kwargs))
#         return original_function(*args, **kwargs)

#     return wrapper

# # 시간 추가
# def my_timer(original_function):  #1
#     import time

#     def wrapper(*args, **kwargs):
#         t1 = time.time()
#         result = original_function(*args, **kwargs)
#         t2 = time.time() - t1
#         print('{} 함수가 실행된 총 시간: {} 초'.format(original_function.__name__, t2))
#         return result

#     return wrapper

# @my_timer
# @my_logger
# def display_info(name, age):
#     time.sleep(1)
#     print('display_info({}, {}) 함수가 실행됐습니다.'.format(name, age))

# display_info("John", 25)