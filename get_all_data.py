import re
import pandas as pd

with open("data/no_context.txt", "r") as f:
	no_context = f.read()
with open("data/with_context.txt", "r") as f:
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

df = pd.read_json("sampled_data.json", lines=True)
with open("full_train.txt", "a") as f:
	for i, row in df.iterrows():
		# answer
		a = row['data']['paragraphs'][0]['qas'][0]['answers'][0]['text']		
		# write the true answer
		f.write(a + "\t" + "[1,1,1]" + "\n")
		
		# append the statements with no context
		for j, answer in enumerate(answers_no_context[i * 5: i * 5 + 5]):
			# case it did print "Answer: ..."
			if "Answer" in answer:
				# strip the "Answer x: " part
				if j == 0:
					# should be true without context
					f.write(answer[10:] + "\t" + "[1,0,0]" + "\n")
				else:
					# false
					f.write(answer[10:] + "\t" + "[0,0,0]" + "\n")
			# case where it didn't have "Answer:..." format
			else:
				if j == 0:
					# should be true without context
					f.write(answer + "\t" + "[1,0,0]" + "\n")
				else:
					# false
					f.write(answer + "\t" + "[0,0,0]" + "\n")

		# append the statements with context
		for j, answer in enumerate(answers_with_context[i * 5: i * 5 + 5]):
			# strip the "Answer x: " part
			# case it did print "Answer: ..."
			if "Answer" in answer:
				if j == 0:
					f.write(answer[10:] + "\t" + "[1,1,0]" + "\n")
				else:
					f.write(answer[10:] + "\t" + "[0,0,0]" + "\n")
			# case where it didn't have "Answer:..." format
			else:
				if j == 0:
					f.write(answer + "\t" + "[1,1,0]" + "\n")
				else:
					f.write(answer + "\t" + "[0,0,0]" + "\n")

df_full = pd.read_csv("data/full_train.txt", sep="\t", header=None)
print(df_full.head(10))