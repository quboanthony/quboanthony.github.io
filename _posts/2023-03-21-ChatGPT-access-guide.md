# ChatGPT 接入指北

## 未来已来
![image-logo1](/img/chatgpt-access-guide/logo1.jpg)
最近一个月，有关ChatGPT以及更加强大的GPT-4的各种报道席卷了的朋友圈，如果你的朋友圈完全没有听说过这个，可能是时候考虑出去多交些朋友了（不是）。介绍GPT相关模型的各种文章可以说已经非常丰富，动动手指就可以搜索到一堆技术与非技术的文章，我也在不断学习当中，建议多少都要看一看。

之所以建议要尽可能了解，是因为根据我目前的理解，以ChatGPT为代表的这一类大型语言模型，其对普通人的最大意义在于生产力工具的进一步跃升（而不是仅仅是什么统治地球消灭人类的博眼球说法）。用较近的例子来比较，的确有点像iphone的横空出世。你可以想象在现在社会你想要与人高效协作，但你就是不用智能机和任何移动端app的情形（强哥除外）。下面是一些我试用ChatGPT的一些例子，可以感受一下：

**根据中文信息和要求生成英文广告slogan**
![image-gpt-example2](/img/chatgpt-access-guide/use-example-02.png)
**根据具体产品需求编写代码示例**
![image-gpt-example1](/img/chatgpt-access-guide/use-example-01.png)

遗憾的是，这么多年过去了，这次“iphone”不但还是美国人先推出的，而且这次也不着急卖给中国人，而是built a wall让你只能远远看着，然后你还只能骂一句nb然后想尽办法gonna pay for it。说人话就是ChatGPT目前还是不对中国等地区开放服务，正常情况下是无法注册直接openai的账号以及使用chatGPT的。

这么多年，看着很多事情都变了，但很多事情也还是没变，可能不是不报还是时辰未到，切不可浮躁。

## DIY注册openai账户

想要把ChatGPT用起来?你需要一个openai账户。
DIY开一个openai账户需要准备的：
- 科学上网环境（注：香港、澳门等都不可用，可以尝试比如美国）。
- 可以接收短信的国外电话号码。
- 邮箱(建议国外邮箱)。


注册流程（全程科学上网环境，ip非大陆、香港、澳门，浏览器无痕模式）
1.进入https://chat.openai.com/auth/login ，点击signup
![image-sign01](/img/chatgpt-access-guide/chatgpt-signup-01.png)
2.填写注册邮箱
![image-sign02](/img/chatgpt-access-guide/chatgpt-signup-02.png)
3.输入密码
![image-sign03](/img/chatgpt-access-guide/chatgpt-signup-03.png)
4.收到verify邮件，登录邮箱verify
![image-sign04](/img/chatgpt-access-guide/chatgpt-signup-04.png)
5.如下页面输入信息（是否真实信息看个人）
![image-sign05](/img/chatgpt-access-guide/chatgpt-signup-05.png)
6.进入发送验证码界面
![image-sign06](/img/chatgpt-access-guide/chatgpt-signup-06.png)
注意这里不能用国内手机号注册，如果有美国手机号可以直接收到code，如果没有可以购买一个临时的虚拟收sms的号码，这里可以使用SMS-Activate这个平台购买虚拟号码，比较方便，支持支付宝，链接如下：

<center> <a href="(https://sms-activate.org/?ref=3098657">SMS-Active链接</a></center>


![image-sign07](/img/chatgpt-access-guide/chatgpt-signup-07.png)

7.点击 <a href="(https://sms-activate.org/?ref=3098657">SMS-Active链接</a>进入网站，注册一个账号。
![image-sign08](/img/chatgpt-access-guide/chatgpt-signup-08.png)

8.注册好账号后，充值余额（余额显示在网页右上角）
![image-sign09](/img/chatgpt-access-guide/chatgpt-signup-09.png)
一次接码OpenAi的验证码费用是大概11卢布，人民币差不多是1块钱，不过只能充美金，最近尝试最低需要充2美金，多充的可以留着以后备用。可以选择支付宝：
![image-sign10](/img/chatgpt-access-guide/chatgpt-signup-10.png)
![image-sign11](/img/chatgpt-access-guide/chatgpt-signup-11.png)

9.充值完成后，在sms-activate主页左侧这里搜索openai相关的虚拟号码，可以看到可以购买的不同国家的虚拟号码，价格也不同
![image-sign12](/img/chatgpt-access-guide/chatgpt-signup-12.png)
点击购买后会出现可以接收短信的号码，**注意号码的有效时间只有20分钟，需要尽快验证，另外可能出现购买的号码输入后错误提示不可用的情况，注意此时我们需要尽早在下边页面点击最右侧x取消接收号码，此时剩余时长的钱会返还给你的账户，我们只需要再换一个号码购买尝试即可，非常方便。**
![image-sign13](/img/chatgpt-access-guide/chatgpt-signup-13.png)

10.openai页面输入购买的号码后，成功的话会显示输入验证码的界面，此时需要等一下，在上面sms-activate的号码界面一会儿会显示发送过来的验证码，复制粘贴该验证码，一般即可注册成功，此时则会进入openai的页面，右上角会显示你的账户。


![image-sign14](/img/chatgpt-access-guide/chatgpt-signup-14.png)

11.确认注册成功后，记得回到sms-activate的号码页面，在原来x的地方会变成一个&#10004;，记得点击一下，剩余时长的钱也会退回保留在你的账户中~

12.注意第10步这里注册成功后跳转的其实是openai api的界面，主要面向developer，如果你想像开篇给的例子一样直接用起来，可以直接进入 https://chat.openai.com/chat 并根据提示再次登录你刚注册好的账户，就可以进入如下界面，开始开心的生产力升级体验之旅~


![image-sign15](/img/chatgpt-access-guide/chatgpt-signup-15.png)

## 写在后面
注册好openai的账户以后，我们就有了体验chatgpt以及其他openai模型的敲门砖，当然openai模型请求输出其实都是收费的，只是费用性价比其实个人感觉很高，当然这个因人而异，请自行把握。另外从页面中可以看到，有升级到PRO的选项，升级为PRO后可以优先使用模型推断的资源，以及优先体验更新的模型（比如GPT-4）。每个账户一开始都是有免费的$18额度，正常试用足够了，但若长期使用，最好绑定一个payment方式来支付相关费用。

当然你肯定猜到了，中国的visa/mastercard卡，都是不能使用的，接下来将专门总结在国内如何绑定可用的支付方式的教程，帮助大家为信仰、啊不生产力充值。