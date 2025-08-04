import sys
from src.logger import logging

def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() ### It tells us about the line in which the error is there.
    file_name=exc_tb.tb_frame.f_code.co_filename ### Its gives us the file name. we habe tb_frame inside exc_tb and insider tb_frame we have f_code and inside that wehave co_filename.
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
