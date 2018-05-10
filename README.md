# 腾讯广告算法大赛2018   
感谢bryan的baselien：https://blog.csdn.net/bryan__/article/details/79623239       
感谢郭大佬的周冠军分享：https://mp.weixin.qq.com/s/Ppw5NICbRs8wJj1ncSdbFw       
感谢葛大佬的周冠军分享：https://mp.weixin.qq.com/s/SVSCbS_df3VW74fcESK3TQ       
感谢yesofcourse的开源：https://github.com/HouJP/kaggle-quora-question-pairs        
感谢大队长第一届比赛的开源：https://github.com/freelzy/Tencent_Social_Ads                  
感谢不知道叫啥名的大佬的平滑代码：https://tianchi.aliyun.com/forum/new_articleDetail.htmlspm=5176.8366600.0.0.2f09311f6RzIy8&raceId=231647&postsId=4844        
主要就是在bryan的baseline上加了几个特征，由于运算资源消耗太大，被公司的小哥哥警告再这样下去直接限制我的资源（本身服务器也一般），加上自己在实习，没有时间和精力做比赛，所以开源。                    
特征方面方面加了三个统计特征，两个是郭大佬分享的，一个是大队长第一届社交广告大赛里的。                         
讲一下代码里的nlp_feature_score函数，这个特征是借鉴yesofcourse kaggle quora解决方案生成特征的函数，针对appIdAction这类特征，原理通过一个例子讲解，例如训练集有两个样本appIdAction_1 = “222 333 444 555 ”  label = 1,appIdAction_2 = “111  333  555 ”  label = 0, 统计特征被split(" ")之后，每个取值的转化率，这块用到贝叶斯平滑，最后appIdAction_1这个样本的特征取值就是连乘(1-p(222))(1-p(333))(1-p(444))(1-p(555)), 第二个样本同理，测试集此特征取值由训练集构造，然后缺失值的样本赋值为-1.0。我只试验了“appIdAction”这个特征，是可以提分的，后面的同类特征也可以逐步加入试验。       

目前线上成绩0.745809(没加appIdAction_score,加上应该能到0.747左右)
