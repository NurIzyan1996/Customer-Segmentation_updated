	9EGr??@9EGr??@!9EGr??@	
ǆA???
ǆA???!
ǆA???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$9EGr??@?O??e??AX9??v?@Y/?$????*     ?W@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?&S???!&???^BC@)?U???؟?1H'?|t@@:Preprocessing2F
Iterator::Modely?&1???!?Kh/?=@)?q??????1 ? ?0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;?O??n??!?0?03@)_?Qڋ?1??2w?,@:Preprocessing2U
Iterator::Model::ParallelMapV2a??+e??!䣓?N>*@)a??+e??1䣓?N>*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?/?$??!{	?%??Q@)??y?):??1`????"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?g??s?u?!?fěo@)?g??s?u?1?fěo@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/n??r?!?)??ԟ@)/n??r?1?)??ԟ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?g??s???!?fěo6@)-C??6j?1q???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9
ǆA???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?O??e???O??e??!?O??e??      ??!       "      ??!       *      ??!       2	X9??v?@X9??v?@!X9??v?@:      ??!       B      ??!       J	/?$????/?$????!/?$????R      ??!       Z	/?$????/?$????!/?$????JCPU_ONLYY
ǆA???b 