	???N@?@???N@?@!???N@?@	FL?D???FL?D???!FL?D???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???N@?@M??St$??A?? ?r?@Y?E???Ԩ?*13333?V@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;?O??n??!??????C@)??????1*?)?@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg??j+???!}}}}}}9@)K?=?U??1??????0@:Preprocessing2F
Iterator::Model??ͪ?Ֆ?!????J8@)g??j+???1}}}}}})@:Preprocessing2U
Iterator::Model::ParallelMapV2?g??s???!4??'@)?g??s???14??'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??ǘ????!??????!@)??ǘ????1??????!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?n??ʱ?!???P?R@)ŏ1w-!?1Vr9?ǎ @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ZӼ?t?!??8??8@)??ZӼ?t?1??8??8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ݓ????!???o?6<@){?G?zd?1;t?W?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9GL?D???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	M??St$??M??St$??!M??St$??      ??!       "      ??!       *      ??!       2	?? ?r?@?? ?r?@!?? ?r?@:      ??!       B      ??!       J	?E???Ԩ??E???Ԩ?!?E???Ԩ?R      ??!       Z	?E???Ԩ??E???Ԩ?!?E???Ԩ?JCPU_ONLYYGL?D???b 