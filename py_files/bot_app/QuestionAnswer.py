import logging
import requests

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import threading
import time

from api import ask_question

class QuestionAnswer(threading.Thread):
    """ 
    Threading subclass to run the question and answer prediction. 
    The current process can take up to 10 seconds.
    """
    def __init__(self, question, response_url):
        super().__init__()
        logging.info('created the QA object')
        self.question = question
        self.response_url = response_url

    def run(self):
        logging.info('running the run() method in the thread')
        logging.info(f'Calling the url {self.response_url}')

        # call the api to generate answers
        results = ask_question(self.question, 5)

        # format the response into json
        counter = 1
        answers = ''
        for a, p in results:
            logging.info(p)
            answers += f'{counter}. {a} \n\tQuestion: {a} \n\tConfidence: {p[0]:.3f}\n'
            counter += 1
        
        json = {
            "text": answers
        }

        # send the message to the post back url
        requests.post(self.response_url, json = json)

    def _format_json(results):
        """
        Formats the results of the ML model into a json response

        Parameters:
        -----------
        results: zip
            Zip of (answer, probability). The probability represents how likely the asked question and 
            the actual question are similar.

        returns: dict
            Returns formatted dictionary of the answers
        """

        return json

