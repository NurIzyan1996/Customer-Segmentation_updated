	ۊ?e?d@ۊ?e?d@!ۊ?e?d@	 ?y??? ?y???! ?y???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ۊ?e?d@?sF????A,Ԛ??@Y?ݓ??Z??*	33333s[@2F
Iterator::ModelZd;?O???!??+z?D@)L7?A`???1?r?3?>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2U0*???!?+z??<@)S?!?uq??1?Ϛ?sh8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM?O???!ԅg?e2@)ŏ1w-!??1??-?˯+@:Preprocessing2U
Iterator::Model::ParallelMapV29??v????!*??G??'@)9??v????1*??G??'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?|a2U??![?*ԅM@)U???N@??1컩>!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zt?! ??7@){?G?zt?1 ??7@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?r?!?DV??@)HP?s?r?1?DV??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZd;?O???!??+z?4@)Ǻ???f?1??<&?f@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?y???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?sF?????sF????!?sF????      ??!       "      ??!       *      ??!       2	,Ԛ??@,Ԛ??@!,Ԛ??@:      ??!       B      ??!       J	?ݓ??Z???ݓ??Z??!?ݓ??Z??R      ??!       Z	?ݓ??Z???ݓ??Z??!?ݓ??Z??JCPU_ONLYY?y???b 