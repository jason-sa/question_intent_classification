
"""
Simple Message class which contains message the bot will respond
"""
class Message(object):
    """
    Instanciates a Message object to create and manage
    Slack onboarding messages.
    """
    def __init__(self):
        super(Message, self).__init__()
        self.channel = ""
        self.timestamp = ""
        self.text = ("Welcome to the AskJason bot!"
                     "\nAsk any question you would like and see if the bot can answer!")
        self.emoji_attachment = {}
        self.pin_attachment = {}
        self.share_attachment = {}
        self.attachments = [self.emoji_attachment,
                            self.pin_attachment,
                            self.share_attachment]
