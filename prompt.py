

photo_emb_prompt = "编码短视频标题，其中<>包裹的是短视频相关标签：{}"
comment_emb_prompt = "编码短视频的评论，其中[]包裹的是表情符号：{}"


seqrec_prompt = []

prompt = {}
prompt["instruction"] = ("在短视频平台上，用户的视频交互历史由一系列按交互时间顺序排列的点赞短视频组成，"
                         "每个短视频包含视频标题和一个相关的热门评论。评论区交互历史则包含多个短视频下的评论交互记录，"
                         "每条记录包含视频标题和用户交互评论。现在，一个用户的短视频交互历史如下：\n{photo_his}\n\n"
                         "评论区交互历史如下：\n{comment_his}\n\n请基于他的短视频交互历史和评论区交互历史记录为他推荐下一个视频：\n")
prompt["response"] = "{response}"
seqrec_prompt.append(prompt)


prompt = {}
prompt["instruction"] = ("在短视频平台上，用户的评论区交互历史包含多个短视频下的评论交互记录，"
                         "每条记录包含视频标题和用户交互评论。视频交互历史则由一系列按交互时间顺序排列的点赞短视频组成，"
                         "每个短视频包含视频标题和一个相关的热门评论。现在，一个用户的评论区交互历史如下：\n{comment_his}\n\n"
                         "短视频交互历史如下：\n{photo_his}\n\n请基于他的评论区交互历史记录和短视频交互历史为他推荐下一个视频：\n")
prompt["response"] = "{response}"
seqrec_prompt.append(prompt)





commrank_prompt = []

prompt = {}
prompt["instruction"] = ("在短视频平台上，用户的视频交互历史由一系列按交互时间顺序排列的点赞短视频组成，"
                         "每个短视频包含视频标题和一个相关的热门评论。评论区交互历史则包含多个短视频下的评论交互记录，"
                         "每条记录包含视频标题和用户交互评论。一个用户的短视频交互历史如下：\n{photo_his}\n\n"
                         "评论区交互历史如下：\n{comment_his}\n\n现在他正在观看视频：{photo}\n"
                         "请根据他的短视频和评论区交互记录以及当前视频内容生成一条其可能交互的评论：\n")
prompt["response"] = "{response}"
commrank_prompt.append(prompt)

prompt = {}
prompt["instruction"] = ("在短视频平台上，用户的评论区交互历史包含多个短视频下的评论交互记录，"
                         "每条记录包含视频标题和用户交互评论。视频交互历史则由一系列按交互时间顺序排列的点赞短视频组成，"
                         "每个短视频包含视频标题和一个相关的热门评论。一个用户的评论区交互历史如下：\n{comment_his}\n\n"
                         "短视频交互历史如下：\n{photo_his}\n\n现在他正在观看视频：{photo}\n"
                         "请根据他的评论区和短视频交互记录以及当前视频内容生成一条其可能交互的评论：\n")
prompt["response"] = "{response}"
commrank_prompt.append(prompt)

# prompt = {}
# prompt["instruction"] = ("在短视频平台上，用户的视频交互历史由一系列按交互时间顺序排列的点赞短视频组成，"
#                          "每个短视频包含视频标题和一个相关的热门评论。评论区交互历史则包含多个短视频下的评论交互记录，"
#                          "每条记录包含视频标题和用户交互评论。一个用户的短视频交互历史如下：\n{photo_his}\n\n"
#                          "评论区交互历史如下：\n{comment_his}\n\n现在他正在观看视频：{photo}，"
#                          "该视频的评论区包含如下评论：\n{candidates}\n\n"
#                          "请根据他的短视频和评论区交互记录以及当前视频内容从评论区挑选一条其可能交互的评论：\n")
# prompt["response"] = "{response}"
# commrank_prompt.append(prompt)

