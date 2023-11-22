import re
import pandas as pd

with open("data/generated/no_context.txt", "r") as f:
	no_context = f.read()
with open("data/generated/with_context.txt", "r") as f:
	with_context = f.read()
	
answers_no_context = [answer for answers in re.findall(r'\<\|assistant\|\>(.*?)<\/s>', no_context, re.DOTALL) for answer in answers.strip().split("\n") 
					  if answer != "" 
					  and not (len(answer) == 9 and "Answer" in answer) 
					  and "    Answer " not in answer 
					  and "<s><|system|>" not in answer 
					  and answer != "    "
					  ]
answers_with_context = [answer for answers in re.findall(r'\<\|assistant\|\>(.*?)<\/s>', with_context, re.DOTALL) for answer in answers.strip().split("\n") 
						if answer != "" 
						and not (len(answer) == 9 and "Answer" in answer) 
						and "    Answer " not in answer 
						and "<s><|system|>" not in answer
						and not (")" in answer and "(" not in answer)
						and any(c.isalpha() for c in answer)
						]

if len(answers_no_context) != 10000:
	raise ValueError("Not enough 'no context' statements")
elif len(answers_with_context) != 10000:
	raise ValueError("Not enough 'with context' statements")

df = pd.read_json("data/sampled_data.json", lines=True)
data_dict = {"sentence": [], "label": []}
for i, row in df.iterrows():
	# answer
	a = row['data']['paragraphs'][0]['qas'][0]['answers'][0]['text']		
	# write the true answer
	data_dict["sentence"].append(a)
	data_dict["label"].append(3)
	
	# append the statements with no context
	for j, answer in enumerate(answers_no_context[i * 5: i * 5 + 5]):
		# case it did print "Answer: ..."
		if "Answer" in answer:
			# strip the "Answer x: " part
			if j == 0:
				# should be true without context
				data_dict["sentence"].append(answer[10:])
				data_dict["label"].append(1)
			else:
				# false
				data_dict["sentence"].append(answer[10:])
				data_dict["label"].append(0)
		# case where it didn't have "Answer:..." format
		else:
			if j == 0:
				# should be true without context
				data_dict["sentence"].append(answer)
				data_dict["label"].append(1)
			else:
				# false
				data_dict["sentence"].append(answer)
				data_dict["label"].append(0)

	# append the statements with context
	for j, answer in enumerate(answers_with_context[i * 5: i * 5 + 5]):
		# strip the "Answer x: " part
		# case it did print "Answer: ..."
		if "Answer" in answer:
			if j == 0:
				data_dict["sentence"].append(answer[10:])
				data_dict["label"].append(2)
			else:
				data_dict["sentence"].append(answer[10:])
				data_dict["label"].append(0)
		# case where it didn't have "Answer:..." format
		else:
			if j == 0:
				data_dict["sentence"].append(answer)
				data_dict["label"].append(2)
			else:
				data_dict["sentence"].append(answer)
				data_dict["label"].append(0)

df_full = pd.DataFrame(data_dict)
df_full.to_csv("data/generated/full_train.csv", index=False)
print(df_full.head(10))