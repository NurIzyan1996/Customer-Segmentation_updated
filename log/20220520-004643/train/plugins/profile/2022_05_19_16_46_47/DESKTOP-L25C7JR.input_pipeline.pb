	5?8EG2@5?8EG2@!5?8EG2@	??FHq?????FHq???!??FHq???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$5?8EG2@s??A???Ax??#??@Y? ?	???*	?????9X@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????Mb??!@h?	?@@)??q????1J??;??;@:Preprocessing2F
Iterator::Model㥛? ???!??N&w?B@)z6?>W??1f?J?ލ;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??d?`T??!5^??x2@)9??v????1?%i???*@:Preprocessing2U
Iterator::Model::ParallelMapV2n????!H??,:$@)n????1H??,:$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?W[?????!|E?و*O@)?q?????1x??\ @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ZӼ?t?!?P^Cy@)??ZӼ?t?1?P^Cy@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicen??t?!H??,:@)n??t?1H??,:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?0?*??!?j?OB5@)??_vOf?1/?re?J@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??FHq???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s??A???s??A???!s??A???      ??!       "      ??!       *      ??!       2	x??#??@x??#??@!x??#??@:      ??!       B      ??!       J	? ?	???? ?	???!? ?	???R      ??!       Z	? ?	???? ?	???!? ?	???JCPU_ONLYY??FHq???b 