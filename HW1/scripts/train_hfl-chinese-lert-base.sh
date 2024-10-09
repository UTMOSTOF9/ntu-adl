method=hfl/chinese-lert-base
bash scripts/_train_paragraph_selection.sh $method
bash scripts/_train_question_answering.sh $method
bash scripts/_test.sh $method