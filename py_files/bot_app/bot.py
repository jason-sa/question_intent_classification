import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import os
import message ## custom class

from slackclient import SlackClient

class Bot(object):
    """ Instanciates a Bot object to handle AskJason app."""
    def __init__(self):
        super(Bot, self).__init__()
        self.name = "askjason"
        self.emoji = ":robot_face:"
        # When we instantiate a new bot object, we can access the app
        # credentials we set earlier in our local development environment.
        self.oauth = {"client_id": os.environ.get("CLIENT_ID"),
                      "client_secret": os.environ.get("CLIENT_SECRET"),
                      # Scopes provide and limit permissions to what our app
                      # can access. It's important to use the most restricted
                      # scope that your app will need.
                      "scope": "bot"}
        self.verification = os.environ.get("VERIFICATION_TOKEN")
        self.token = os.environ.get("BOT_TOKEN")

        # NOTE: Python-slack requires a client connection to generate
        # an oauth token. We can connect to the client without authenticating
        # by passing an empty string as a token and then reinstantiating the
        # client with a valid OAuth token once we have one.
        self.client = SlackClient("")

        # Use the dictionary to log which user the bot has interacted with
        self.users = set()

    def welcome_message(self, channel):
        """
        Create and send a welcome message to new users. 

        Parameters
        ----------
        channel : str
            id of the Slack channel associated with the incoming message
        """
        message_obj = message.Message()
        post_message = self.client.api_call("chat.postMessage",
                                            channel=channel,
                                            username=self.name,
                                            icon_emoji=self.emoji,
                                            text=message_obj.text,
                                            token=self.token
                                            )
        logging.info(f'welcome_message: {post_message}')