	X?5?;?@X?5?;?@!X?5?;?@	????&??????&??!????&??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$X?5?;?@?5^?I??AyX?5?{@YP??n???*?????LZ@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeataTR'????!???w?>@)??q????1Bd_??9@:Preprocessing2F
Iterator::Model??y?):??!Ic??@@)
ףp=
??1?/? Ic5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??@??ǘ?!!?/? 7@)lxz?,C??1?$?Y<*@:Preprocessing2U
Iterator::Model::ParallelMapV2???<,Ԋ?!?ĚW??(@)???<,Ԋ?1?ĚW??(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?5?;Nѱ?![?t8?P@)??_?L??1L>@Ҙ?#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_?L??!L>@Ҙ?#@)??_?L??1L>@Ҙ?#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOv?!OjF?@)??_vOv?1OjF?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??q????!Bd_??9@)??_vOf?1OjF?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????&??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?5^?I???5^?I??!?5^?I??      ??!       "      ??!       *      ??!       2	yX?5?{@yX?5?{@!yX?5?{@:      ??!       B      ??!       J	P??n???P??n???!P??n???R      ??!       Z	P??n???P??n???!P??n???JCPU_ONLYY????&??b 