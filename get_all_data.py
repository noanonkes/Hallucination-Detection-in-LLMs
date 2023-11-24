import pandas as pd

if __name__ == "__main__":
	with open("data/generated/no_context.txt", "r") as f:
		no_context = f.readlines()
	with open("data/generated/with_context.txt", "r") as f:
		with_context = f.readlines()
		
	if len(no_context) != 10000:
		raise ValueError("Not enough 'no context' statements")
	elif len(with_context) != 10000:
		raise ValueError("Not enough 'with context' statements")

	# to create a csv file with text + label
	df = pd.read_json("data/sampled_data.json", lines=True)
	data_dict = {"sentence": [], "label": []}
	for i, row in df.iterrows():
		# the true answer
		answer = row['data']['paragraphs'][0]['qas'][0]['answers'][0]['text']		
		data_dict["sentence"].append(answer)
		data_dict["label"].append(3)
		
		# append the statements with no context
		data_dict["sentence"].extend(no_context[i * 5: i * 5 + 5])
		data_dict["label"].extend([1, 0, 0, 0, 0])

		# append the statements with context
		data_dict["sentence"].extend(with_context[i * 5: i * 5 + 5])
		data_dict["label"].extend([2, 0, 0, 0, 0])

	df_full = pd.DataFrame(data_dict)
	df_full.to_csv("data/generated/full_answers.csv", index=False)