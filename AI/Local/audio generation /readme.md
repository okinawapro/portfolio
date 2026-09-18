1. Needs python and git

2. Download huggingface musicgen-small
git clone https://huggingface.co/facebook/musicgen-small

3. Install huggingface transformers
pip install git+https://github.com/huggingface/transformers.git
pip install torch soundfile scipy

4. this should be enough. run the generate_local_music.py program

Duration	Recommended max_new_tokens
10 sec	256 tokens
30 sec	768 tokens
60 sec	1536 tokens

small model will break if over 2000 tokens are used.
