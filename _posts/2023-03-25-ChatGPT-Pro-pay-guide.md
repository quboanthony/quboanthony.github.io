# ChatGPT 充值指北

## Pro Style

上一篇《ChatGPT接入指北》，总结了如何绕过限制，access openai的服务。当然光光解决了贫困问题还不够，人民还有追求进一步提升物质文化水平的需求。这里openai作为一个半盈利组织，“不负众望”地给大家提供一个充值升级为Plus会员的机会，每月20刀。充值后可以优先获得模型计算的资源，以及优先新模型（e.g. GPT-4）的使用资格，只能说高低我得冲一波看看。

![image-funny](/img/chatgpt-pay-guide/funny-01.jpg)

在openai提供的[ChatGPT](https://chat.openai.com/chat)的测试交互界面，如果你的ip是美国或者欧洲的，可以看到一个 “Upgrade to Plus”的选项：

![image-5](/img/chatgpt-pay-guide/chatgpt-pay-05.png)

点击后，我们可以看到ChatGPT Plus的价格，以及可以享受到的“特权”：

![image-6](/img/chatgpt-pay-guide/chatgpt-pay-06.png) 

点击Upgrade plan，弹出页面后，看到我们需要提供支付方式和账单地址。

![image-9](/img/chatgpt-pay-guide/chatgpt-pay-09.png) 

这里需要**注意区分**的是，上边页面升级的才是ChatGPT这个模型的订阅会员，如果你在登录后的是下面这个[页面](https://platform.openai.com/overview)中，这个我们也可以看到一个“Upgrade”选项，**但区别是，这个是针对developer的API接口的支付方式绑定选项**， 是当你希望作为一个程序开发人员，通过API的方式以token数计费，调用openai的各种模型输出时的支付绑定入口。而上边的ChatGPT相当于一个单独的会员订阅，所以**两者是独立的**，要强调是因为这里API绑定支付方式后会预支5刀的费用，且跟上边的ChatGPT Plus无关，故只想在上边界面使用ChatGPT，不需要API的，则不要在这里绑定：

![image-1](/img/chatgpt-pay-guide/chatgpt-pay-01.png)


点开后，可以看到Set up paid account的选项，点击并选择你是individual（个人）还是behalf of a company(代表公司)。

![image-2](/img/chatgpt-pay-guide/chatgpt-pay-02.png)

![image-3](/img/chatgpt-pay-guide/chatgpt-pay-03.png)

然后就会弹出输入信用卡信息的界面，这里需要填写美国的信用卡和美国的账单地址，直接绑定即可。

![image-4](/img/chatgpt-pay-guide/chatgpt-pay-04.png)

**问题来了：无论是ChatGPT Plus订阅， 还是API调用的支付，这里国内的实体VISA、MasterCard，以及国内的账单地址，都是绑定不上的，必须美国信用卡和账单地址的才行。**

为了成功绑定支付方式，目前了解的解决办法有

1. 通过获得ITIN号码（个人报税识别号码）的方式申请真实的美国信用卡，需要护照，且填写各类表格，成本460刀左右，之后每个月20刀的美国私人地址成本。好处是可以申领到一张真实的美国信用卡，可以开始积累信用，劣势是申请周期比较长。（后续也要好好研究下）

    - [参考链接](https://www.zhihu.com/question/49681885)

2. 申请一个**虚拟Visa信用卡+数字货币平台**。优势1是周期短，本人尝试大约1小时以内搞定，全部在手机和电脑上操作；2成本相对低，在各种转账时会有转账成本，但用于openai平台的小额充值，远低于上边几百刀的成本（例如我以174rmb购买了25刀的USDT，从开卡刀最后转到卡里还有22.96刀，总成本为一点汇率差+2.04刀各类手续费，当然不同操作时间场景下具体金额会有差异，但量级不会差很多）。劣势是需要在数字货币平台以及虚拟信用卡平台操作，有一定学习成本和操作门槛。

这里我主要分享下我实践成功的第2种方式的手把手流程，我用的是欧易+Depay的组合（可能是目前国内唯一的方案组合），注意下面操作也需要在可以访问ChatGPT的上网环境下进行。

- [比较好用的科学上网梯子](http://kingfast.info/index.php/index/register/?yqi=63180)
- 查看当前ip的方式：http://en.ipip.net/

## 注册1个数字货币交易平台

这里注册[欧易](https://okx.com/join/12215847)的账号，注册不需要科学上网。经过之前的整顿调整，暂时貌似目前国内比较安全的只有欧易了，目前已使用了有一段时间了感觉也还是可以。只是这里要说一下：**不要炒币，不要炒币，不要炒币！**

之所以需要注册数字货币交易平台，是因为接下来我们要使用的虚拟信用卡平台Depay不接受人民币直接充值，只接受欧元、美元和USDT等。所以需要在数字货币交易平台将人民币兑换为USDT，再充值到Depay上兑换为美元。

所以首先去[官网注册欧易账号](https://okx.com/join/12215847)。账号注册好后，可以去下载安装APP方便手机操作使用，如果你是苹果手机，则使用一个海外的AppleID就可以轻松下载欧易的app。

接下来就是兑换了，在app中登录账户后，点击资产--充币--法币充值，并按照提示购买少量额度的USDT，除了汇率损失之外，手续费到Depay再兑换USD下来各种成本可能要2USDT左右，故尽量留出稍多些的使用余量，一方面不要单次购买太大额的数字货币，一方面也有要尽量保证到Depay后足够订阅ChatGPT，尽量减少兑换和转账的次数控制成本。

![image-5](/img/chatgpt-pay-guide/okc-01.jpg)

![image-6](/img/chatgpt-pay-guide/okc-02.jpg)

![image-7](/img/chatgpt-pay-guide/okc-03.jpg)

购买好USDT后，我们就可以开始准备Depay平台的注册。

## 注册申请一张虚拟信用卡

Depay是支持数字货币交易的虚拟信用卡，可以支持在全世界范围内消费，甚至可以绑定微信、支付宝消费。这里我们需要注册并开通一张虚拟信用卡，用来支付ChatGPT的订阅。

在安全性上，Depay持有的是美国MSB牌照，并接受Fincen(金融犯罪执法局)的合规监管，故可以为提供KYC认证的申请人合法开通信用卡，但如果对该新生平台仍有一些风险担忧，最佳的方式是不要一次转入太多的USDT，小额够用就好，openai的订阅需要的金额也不大。

[Depay账户注册链接](https://depay.depay.one/web-app/register-h5?invitCode=669571&lang=en-us) ，或扫描下方二维码注册：

![image-8](/img/chatgpt-pay-guide/depay-01.jpg)

注册好账户后，现在可以在apple store下载官方的app，并在app中申请开卡：

![image-9](/img/chatgpt-pay-guide/depay-02.jpg)

点开后首先可以选择需要完成KYC的方式（0 USDT），或者不需要KYC的方式（10 USDT）, 若想长期使用可以使用需要KYC的方式，手续费也更便宜。在选择需要KYC之后，又会有几种开卡选项，简单说就是又额外提供了10USDT、50USDT开卡费的套餐，可以通过付一笔钱的方式获得更低或减免月租费用，以及更低的手续费。我这里就选择了一路0 USDT的方式，KYC认证，等待5-10min认知成功后，便获得了你的第一张虚拟信用卡。

![image-10](/img/chatgpt-pay-guide/depay-04.png)

## 从数字货币平台将USDT转账到Depay

到目前为止，我们需要做的，就是将兑换好的USDT转账到Depay。这里需要注意的是，转账走的是数字货币的方式，并不是任何中央银行，所以主要要确认好充值的地址与是使用的公链名称！

具体来说，首先在Depay app中，找到钱包，选择USDT，然后选择充币的公链。这里正常Polygon手续费会更便宜一点，这个可以在欧易上操作时看到具体的手续费。

![image-11](/img/chatgpt-pay-guide/depay-05.jpg)

选择完毕后，Depay就会生成接收转账的充值地址与对应的二维码，复制充值地址。

接下来打开欧易app，从资产--提币--选择USDT--选择链上提币--然后将复制的充值地址粘贴到页面中，就可以选择提币的网络了，因为之前在Depay上时通过Polygon生成的充值地址，这里也要选择Polygon（一定要对应，要不转出去找不回来的）。

![image-12](/img/chatgpt-pay-guide/depay-06.jpg)

提交再验证后，便完成了转账（这里欧易一般有1min的免费取消期限，如果发现有什么明显的问题的话）。

之后便是等到USDT到账Depay了，这里可能需要等待5-10min的时间完全到账，如果公链选择是匹配的，在Depay app中是可以看到在途的USDT记录的。

到账后，在Depay中点击钱包，需要将USDT都兑换成美元。

![image-13](/img/chatgpt-pay-guide/depay-07.jpg)

兑换后选择app首页的to card/充值，将美元存入卡中，就正式完成了Depay的充值了！

![image-14](/img/chatgpt-pay-guide/depay-08.jpg)

注意：

- Depay的虚拟信用卡没有透支额度，需要先充值，再消费，否则会有余额不足交易失败。

- 信用卡的CVV码需要在app首页通过认证获得，记得包括卡号在内，不要随便透露给任何人。

## 最后一步

现在我们的虚拟信用卡里，已经有了可以消费的额度，最后一步就是给ChatGPT Plus的订阅支付页面填写信用卡信息了。

这里还需要的是

1. 科学上网环境（一定要是美国或欧洲的ip）。
2. 无痕网页，清除cookie，若一直可以正常登录ChatGPT那不用清除也问题不大。
3. 搜索美国地址生成器，去生成免税州的一个美国地址作为账单地址，具体哪几个州免税可以自己搜下。
4. ChatGPT Plus的订阅支付页面，主要需要填写的就是卡号、日期、CVV，以及美国账单地址与邮编等。

支付成功后，就会获得这个“尊贵”的标识


![image-15](/img/chatgpt-pay-guide/chatgpt-pay-10.png)

恭喜你！高级的生产力正在向你招手，赶紧用起来吧！
