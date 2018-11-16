from chatterbot import ChatBot
chatbot = ChatBot('Brandon', trainer='chatterbot.trainers.ListTrainer')
chatbot.train([
    "Hello",
    "Hi there!",
    "How are you doing",
    "I'm doing great, how about you",
    "That is good to hear",
    "Thank you",
    "You're welcome",
    "What is your name",
    "My name is Chitty",
])
chatbot.train([
    "Good bye!",
    "See you soon!",
    "Sorry, I don't know about this.."
])