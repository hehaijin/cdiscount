
split_percent= 0.6
#generate mini-batch for deep learning

def show():
	"""
	Save a few images from the TRAIN_FILE.
	For more elaborate visualizations and statistics, see other kernels such as
	<https://www.kaggle.com/bguberfain/just-showing-a-few-images>
	<https://www.kaggle.com/bguberfain/naive-statistics>
	Run with `cxflow dataset show cdc`
	"""
	os.makedirs(sampledir, exist_ok=True)
	data = bson.decode_file_iter(open(path.join(data_root, train_file), 'rb'))
	for i, example in takewhile(lambda x: x[0] < 10, enumerate(data)):
		for j, image in enumerate(example['imgs']):
			with open(path.join(sampledir, 'image_{}_{}.jpg'.format(i, j)), 'wb') as file_:
				file_.write(image['picture'])
	logging.info('Images saved to `%s`', sampledir)
	
	
	
def split(self):
	"""
	Split train data to train and validation sets and compute (category_id -> integer class) mapping.
	Run with `cxflow dataset split cdc`
	:return:
	"""
	# read example headers
	logging.info('Reading examples metadata, this may take a minute or two')
	ids = []
	categories = []
	for example in bson.decode_file_iter(open(path.join(data_root,train_file), 'rb')):
		ids.append(example['_id'])
		categories.append(example['category_id'])

	# generate random split
	size = len(ids)
	train_size = int(size*(1-split_percent))
	valid_size = (size-train_size)
	split = ['train']*train_size + ['valid']*valid_size
	random.shuffle(split)

	# save split
	split_df = pd.DataFrame({'id': ids, 'split': split})
	split_path = path.join(self._data_root, self._split_file)
	split_df.to_csv(split_path, index=False)
	logging.info('Split train-valid of size %s-%s was written to `%s`', train_size, valid_size, split_path)

	# save (category_id -> integer class) mapping
	categories = sorted(list(set(categories)))
	categories_df = pd.DataFrame({'category_id': categories, 'class': list(range(len(categories)))})
	categories_path = path.join(self._data_root, self.CATEGORIES_FILE)
	categories_df.to_csv(categories_path, index=False)
	logging.info('Categories mapping saved to `{}`'.format(categories_path))

def categoryids():
	df=pd.read_csv(path.join(dataroot, 'category_names.csv'))
	df.reset_index()
	#df.columns[2] = 'New_ID'
	df['value'] = df.index
	return df
	
