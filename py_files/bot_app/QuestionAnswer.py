import logging
import requests
import threading
from api import ask_question

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class QuestionAnswer(threading.Thread):
    """ 
    Threading subclass to run the question and answer prediction. 
    The current prediction process can take up to 10 seconds.
    """
    def __init__(self, question, response_url):
        """
        Initalizes

        Parameters:
        ----------

        question: string
            Question to look-up similar questions
        
        response_url: string
            Response URL provided by Slack API to send the results back once completed
        """
        super().__init__()
        logging.info('created the QA object')
        self.question = question
        self.response_url = response_url

    def run(self):
        logging.info('running the run() method in the thread')
        logging.info(f'Calling the url {self.response_url}')

        # call the api to generate answers
        results = ask_question(self.question, 3)

        # format the response into json
        counter = 1
        answers = ''
        for a, p in results:
            logging.info(p)
            answers += (f'{counter}. *Answer:* {a[0]} \n\t' + 
                       f'*Confidence:* {p[0]*100:.1f}%\n\t' + 
                       f'*Simiar Question:* {a[1]} \n\n')
            counter += 1
        
        json = {
            "text": answers
        }

        # send the message to the post back url
        requests.post(self.response_url, json = json)