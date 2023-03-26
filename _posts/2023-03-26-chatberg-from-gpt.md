# 基于Openai接口的聊天机器人

基于python，openai api和gradio构建一个聊天机器人网页，这个网页需要支持用户注册，且需要有给注册用户管理api token使用的功能：包括充值token，对api消费的token进行计数，并在用户界面进行展示。

```python
import openai
import gradio as gr
from flask import Flask, request, jsonify

# OpenAI API credentials
openai.api_key = "YOUR_API_KEY_HERE"

# Flask app initialization
app = Flask(__name__)

# User management functions
users = {}

def register_user(username):
    users[username] = {"tokens": 0}

def recharge_tokens(username, amount):
    users[username]["tokens"] += amount

def consume_token(username):
    users[username]["tokens"] -= 1

# Chatbot function
def chatbot(query):
    # check if user is registered
    if "username" not in request.cookies:
        return "Please register first."
    username = request.cookies["username"]
    # check if user has enough tokens
    if users[username]["tokens"] < 1:
        return "You don't have enough tokens. Please recharge."
    # consume one token
    consume_token(username)
    # call OpenAI API
    response = openai.Completion.create(
        engine="davinci",
        prompt=query,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

# Flask routes
@app.route("/")
def index():
    return gr.Interface(fn=chatbot, inputs="textbox", outputs="textbox")()

@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username")
    if username in users:
        return jsonify({"message": "Username already exists."}), 400
    register_user(username)
    response = jsonify({"message": "User registered successfully."})
    response.set_cookie("username", username)
    return response

@app.route("/recharge", methods=["POST"])
def recharge():
    username = request.cookies["username"]
    amount = int(request.form.get("amount"))
    recharge_tokens(username, amount)
    return jsonify({"message": "Tokens recharged successfully."})

@app.route("/tokens", methods=["GET"])
def tokens():
    username = request.cookies["username"]
    return jsonify({"tokens": users[username]["tokens"]})

# Flask app run
if __name__ == "__main__":
    app.run()
```

在这个代码示例中，我们使用了OpenAI的API来实现聊天机器人的功能，使用了Gradio库来实现网页界面，使用了Flask来实现用户注册和Token管理的接口。

具体来说，我们定义了以下几个函数：

- `register_user(username)`: 注册一个新用户，将其添加到`users`字典中。
- `recharge_tokens(username, amount)`: 为指定用户充值Token。
- `consume_token(username)`: 消费一个Token，即将用户的Token数减1。
- `chatbot(query)`: 聊天机器人功能的实现函数。在这个函数中，我们首先检查用户是否已经注册，然后检查用户是否有足够的Token，如果有，就调用OpenAI API来获取聊天机器人的回复。最后，我们消费一个Token并返回回复。
- `index()`: Gradio界面的实现函数。我们在这里创建了一个输入框和输出框，并将其绑定到`chatbot()`函数上。
- `register()`: 用户注册接口。我们在这里从POST请求的form中获取用户名，检查是否已经存在，如果不存在，则注册新用户。注册成功后，我们将用户名保存到cookie中。
- `recharge()`: Token充值接口。我们从POST请求的form中获取充值的Token数量，并将其添加到指定用户的Token数中。
- `tokens()`: 查询当前用户的Token数。

在代码中，我们使用了一个`users`字典来保存所有已注册用户的信息。每个用户都拥有一个`tokens`属性来保存其当前的Token数。当用户消费Token时，我们会调用`consume_token()`函数来将其Token数减1。如果用户没有足够的Token，我们将返回一个错误信息。

最后，我们在代码末尾使用`app.run()`来启动Flask应用程序。你可以将其保存为一个Python脚本，并在命令行中执行`python your_script.py`来运行它。