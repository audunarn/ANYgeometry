"""Display-only cap geometry: explicit volume, no engineering edits."""
import numpy as np
import pytest
from anygeometry.volume_cut import BoxVolume,VolumeCutError
from anygeometry.display_caps import box_volume_cut_caps


def test_box_caps_rotate_clip_and_preserve_source():
    volume=BoxVolume((-1,-1,-1),(1,1,1));before=volume.to_dict()
    caps=box_volume_cut_caps(volume,[((1,0,0),0),((0,1,0),0)])
    assert {cap.plane_index for cap in caps}=={0,1}
    for cap in caps:
        assert np.all(cap.vertices[:,0]>=0) and np.all(cap.vertices[:,1]>=0)
        normal=np.cross(cap.vertices[1]-cap.vertices[0],cap.vertices[2]-cap.vertices[1])
        assert normal[cap.plane_index]<0
        assert cap.vertices.shape==(4,3)
    assert volume.to_dict()==before
    assert box_volume_cut_caps(volume,[((1,0,0),2)])==()
    assert box_volume_cut_caps(volume,[((1,0,0),1)])==()
    assert box_volume_cut_caps(volume,[((1,0,0),-1)])==()
    assert box_volume_cut_caps(volume,[((1e300,0,0),0)])[0].vertices.shape==(4,3)
    with pytest.raises(VolumeCutError):box_volume_cut_caps(object(),[])
    with pytest.raises(VolumeCutError):box_volume_cut_caps(volume,[((0,0,0),0)])
    with pytest.raises(VolumeCutError):box_volume_cut_caps(volume,[((1,0,0),2),((0,0,0),0)])


def test_translated_rotated_closed_operand():
    matrix=np.eye(4);matrix[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];matrix[:3,3]=[1e6,2e6,3e6]
    volume=BoxVolume((-1,-2,-3),(1,2,3),tuple(map(tuple,matrix)))
    caps=box_volume_cut_caps(volume,[((1,0,0),1e6)])
    assert len(caps)==1
    np.testing.assert_array_equal(caps[0].vertices[:,0],np.repeat(1e6,4))
