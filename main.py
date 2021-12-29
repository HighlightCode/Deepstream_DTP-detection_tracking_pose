import sys
sys.path.append('../')
sys.path.append('/usr/lib/python3/dist-packages')
import gi
from optparse import OptionParser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from utils.utils import Reverse_scale
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.kbhit import KBHit
from common.FPS import GETFPS
import configparser

import ctypes
import numpy as np 

import pyds

PGIE_CLASS_ID_PERSON = 2
past_tracking_meta=[0]


Reverse_scale=Reverse_scale((224,224))
OUTPUT_VIDEO_NAME = "./result/walking.mp4"

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_PERSON:0,
    }
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list

        # set the display_meta to display numpy array on nveglsink
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_circles = 16
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.class_id==2:
                store_frame = True
                obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)
                meta_l = obj_meta.tracker_bbox_info.org_bbox_coords.left
                meta_t = obj_meta.tracker_bbox_info.org_bbox_coords.top
                meta_w = obj_meta.tracker_bbox_info.org_bbox_coords.width
                meta_h = obj_meta.tracker_bbox_info.org_bbox_coords.height
                #print(meta_l, " ", meta_t, " ", meta_w, " ", meta_h)
                #print(obj_meta.rect_params.left, " " , obj_meta.rect_params.top)
                l_user_meta = obj_meta.obj_user_meta_list
                while l_user_meta:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                    meta_type = user_meta.base_meta.meta_type
                    if meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                        meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        #print(meta.num_output_layers) -> if you have more than 1 layer you should iterate the num_output_layers
                        layer = pyds.get_nvds_LayerInfo(meta, 0)
                        ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                        v = np.ctypeslib.as_array(ptr, shape=(16,2))
                        v = ((v + 1) * np.array([224,224])-1)/2
                        Reverse_scale((meta_l, meta_t, meta_w, meta_h),v)
                        
                        circles = 0
                        for i in range(0,16):
                            cparams = display_meta.circle_params[i]
                            cparams.xc = int(v[i][0])
                            cparams.yc = int(v[i][1])
                            cparams.radius = 2
                            if 0 <= i < 6:
                                cparams.circle_color.set(0, 67, 54, 1)
                            elif 6 <= i < 8:
                                cparams.circle_color.set(0, 254, 0, 1)
                            elif 8 <= i < 10:
                                cparams.circle_color.set(254, 0, 0, 1)
                            else: 
                                cparams.circle_color.set(0, 0, 255, 1)
                            circles+=1
                        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
                        #print(v)
                    try:
                        l_user_meta = l_user_meta.next
                    except StopIteration:
                        break
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
            # obj_counter[obj_meta.class_id] += 1

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Person_count={}".format(frame_number, num_rects,obj_counter[PGIE_CLASS_ID_PERSON])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    #past traking meta data
    if(past_tracking_meta[0]==1):
        l_user=batch_meta.batch_user_meta_list
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting is done by pyds.NvDsUserMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone
                user_meta=pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
                try:
                    # Note that user_meta.user_meta_data needs a cast to pyds.NvDsPastFrameObjBatch
                    # The casting is done by pyds.NvDsPastFrameObjBatch.cast()
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone
                    pPastFrameObjBatch = pyds.NvDsPastFrameObjBatch.cast(user_meta.user_meta_data)
                except StopIteration:
                    break
                for trackobj in pyds.NvDsPastFrameObjBatch.list(pPastFrameObjBatch):
                    print("streamId=",trackobj.streamID)
                    print("surfaceStreamID=",trackobj.surfaceStreamID)
                    for pastframeobj in pyds.NvDsPastFrameObjStream.list(trackobj):
                        print("numobj=",pastframeobj.numObj)
                        print("uniqueId=",pastframeobj.uniqueId)
                        print("classId=",pastframeobj.classId)
                        print("objLabel=",pastframeobj.objLabel)
                        for objlist in pyds.NvDsPastFrameObjList.list(pastframeobj):
                            print('frameNum:', objlist.frameNum)
                            print('tBbox.left:', objlist.tBbox.left)
                            print('tBbox.width:', objlist.tBbox.width)
                            print('tBbox.top:', objlist.tBbox.top)
                            print('tBbox.right:', objlist.tBbox.height)
                            print('confidence:', objlist.confidence)
                            print('age:', objlist.age)
            try:
                l_user=l_user.next
            except StopIteration:
                break
    return Gst.PadProbeReturn.OK

def main(args):

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)
    
    # Create gstreamer elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    
    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    # element for playing the mp4 file 
    print("Creating Qtdemux \n")
    demux = Gst.ElementFactory.make("qtdemux", "demux")
    if not demux:
        sys.stderr.write(" Unable to create demux plugin \n")

    # add demux - video_queue function
    def handle_demux_pad_added(src, new_pad, *args, **kwargs):
        if new_pad.get_name().startswith('video'):
            print("\n--- demux video pad added:", new_pad, new_pad.get_name())
            queue_sink_pad = video_queue.get_static_pad('sink')
            new_pad.link(queue_sink_pad)
        else:
            print("\n--- demux * pad added:", new_pad, new_pad.get_name())

    demux.connect("pad-added", handle_demux_pad_added)

    print("Creating Queue \n")
    video_queue = Gst.ElementFactory.make("queue", "video_queue")
    if not video_queue:
        sys.stderr.write(" Unable to create video_queue plugin \n")
        
    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    print("Creating H264Parser \n")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write(" Unable to create h264 parser \n")

    # Use nvdec_h264 for hardware accelerated decode on GPU
    print("Creating Decoder \n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    # Seconday inference for pose Estimation
    sgie = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    if not sgie:
        sys.stderr.write(" Unable to create secondary_gie \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    if store_true:
        print("Creating Fake sink \n")
        queue = Gst.ElementFactory.make("queue", "queue")
        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
        encoder = Gst.ElementFactory.make("avenc_mpeg4", "encoder")
        codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")
        container = Gst.ElementFactory.make("qtmux", "qtmux")
        sink = Gst.ElementFactory.make("filesink", "filesink")
        # sink = Gst.ElementFactory.make("fakesink", "fakesink")
        if not sink:
            sys.stderr.wrtie(" Unable to create fake sink \n")
    else:
        if is_aarch64():
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        print(" Create EGL SINK \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    print("Playing file %s " %input_file)
    source.set_property('location', input_file)
    streammux.set_property('width', 1080)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000) #4000000-> 30 fps
    pgie.set_property('config-file-path', "./dstest1_pgie_caffe_config.text")
    sgie.set_property('config-file-path', "./dstest1_sgie_config.text")

    # SET PROPERTY's of TRACKER 
    config = configparser.ConfigParser()
    config.read('./tracker/test_nvdcf_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)

    if store_true:
        encoder.set_property("bitrate", 3000000)
        sink.set_property("location", OUTPUT_VIDEO_NAME)
        sink.set_property("sync", 0)
        sink.set_property("async", 0)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(demux)
    pipeline.add(video_queue)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    if store_true:
        pipeline.add(queue)
        pipeline.add(nvvidconv2)
        pipeline.add(encoder)
        pipeline.add(codeparser)
        pipeline.add(container)
    pipeline.add(sink)
    if is_aarch64() and not store_true:
        pipeline.add(transform)

    # we link the elements together
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(demux)
    #demux.link(video_queue)
    
    #queue_sink = video_queue.get_static_pad("sink")
    #demux_pad = demux.get_pad("src")
    #demux_pad.link(queue_sink)

    video_queue.link(h264parser)
    h264parser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie)
    sgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64() and not store_true:
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(queue)
        queue.link(nvvidconv2)
        nvvidconv2.link(encoder)
        encoder.link(codeparser)
        codeparser.link(container)
        container.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)


    # ADD GObject.timeout_add() function to add time out .
    # GObject.timeout_add(4000, event_thread_func, pipeline)
    
    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

def parse_args():
    parser = OptionParser()

    parser.add_option("-i", "--input-file", dest="input_file",
                      help="Set the input mp4 file", metavar="FILE")

    parser.add_option("-s", "--store", action="store_true",
                      dest="store", default=False,
                      help="no display & store frame")

    parser.set_defaults(feature=False)

    (options, args) = parser.parse_args()

    global input_file
    global store_true

    input_file = options.input_file
    store_true = options.store
    print(f"store true option : {store_true}")

    if not input_file:
        print("Usage: python3 main.py -i <file_name> -s <store_true_option>")
        return 1

if __name__ == '__main__':
    ret = parse_args()
    # If argument parsing fails, returns failure (non-zero)
    if ret == 1:
        sys.exit(1)
    sys.exit(main(sys.argv))
