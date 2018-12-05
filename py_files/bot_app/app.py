# -*- coding: utf-8 -*-
"""
A routing layer for the onboarding bot tutorial built using
[Slack's Events API](https://api.slack.com/events-api) in Python
"""
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('Starting app...')

import json
import bot
from flask import Flask, request, make_response, render_template
import requests
import time

from QuestionAnswer import QuestionAnswer

pyBot = bot.Bot()
slack = pyBot.client

app = Flask(__name__)

def _event_handler(event_type, slack_event):
    """
    A helper function that routes events from Slack to our Bot
    by event type and subtype.
    Parameters
    ----------
    event_type : str
        type of event recieved from Slack
    slack_event : dict
        JSON response from a Slack reaction event
    Returns
    ----------
    obj
        Response object with 200 - ok or 500 - No Event Handler error
    """
    user_id = slack_event["event"].get("user")
    
    # bots do not have a user_id, and do not want to respond
    if user_id is None: 
        pyBot.users.add(user_id)

    channel = slack_event["event"]["channel"]

    # ================ Message Events =============== #
    logging.info(f'Event type: {event_type}')
    
    # Send a welcome message if this is the first message recieved by a user
    if event_type == "message" and user_id not in pyBot.users: 
        # Send the welcome message
        pyBot.welcome_message(channel)
        pyBot.users.add(user_id)
        return make_response("Welcome Message Sent", 200,)

    # All other messages are expected to be questions
    elif event_type == "message" and user_id is not None:
        logging.info(f'Question asked by user: {user_id}')
        # send thinking message
        pyBot.send_thinking(channel)

        # send top n answers
        user_text = slack_event["event"]["text"]
        pyBot.answer_question(user_text, channel, 3)

        return make_response("Question Answered", 200,)

    # ============= Event Type Not Found! ============= #
    # If the event_type does not have a handler
    message = "You have not added an event handler for the %s" % event_type
    # Return a helpful error message
    return make_response(message, 200, {"X-Slack-No-Retry": 1})

# Route which handles the slash command (/q)
@app.route("/question", methods=["GET","POST"])
def answer_question():

    logging.info('Answering a question...')

    url = request.values['response_url']
    question = request.values['text']

    # spawn a thread to post back the results of the predicted answers
    question_answerer = QuestionAnswer(question, url)
    question_answerer.start()

    return make_response(f'Thinking...{question}', 200, {'X-Slack-NoRetry': 1})

@app.route("/listening", methods=["GET", "POST"])
def hears():
    """
    This route listens for incoming events from Slack and uses the event
    handler helper function to route events to our Bot.
    """
    logging.info('Listening request...')
    slack_event = json.loads(request.data)
    logging.info(f'Slack event: {slack_event}')

    # ============= Slack URL Verification ============ #
    # In order to verify the url of our endpoint, Slack will send a challenge
    # token in a request and check for this token in the response our endpoint
    # sends back.
    #       For more info: https://api.slack.com/events/url_verification
    if "challenge" in slack_event:
        logging.info('Challenge event')
        return make_response(slack_event["challenge"], 200, {"content_type":
                                                             "application/json"
                                                             })

    # ====== Process Incoming Events from Slack ======= #
    # If the incoming request is an Event we've subcribed to
    if "event" in slack_event and "bot_id" not in slack_event:
        logging.info('Slack event....')
        event_type = slack_event["event"]["type"]
        # Then handle the event by event_type and have your bot respond
        return _event_handler(event_type, slack_event)

    # If our bot hears things that are not events we've subscribed to,
    # send a quirky but helpful error response
    return make_response("[NO EVENT IN SLACK REQUEST] These are not the droids\
                         you're looking for.", 404, {"X-Slack-No-Retry": 1})


if __name__ == '__main__':
    app.run(debug=True)