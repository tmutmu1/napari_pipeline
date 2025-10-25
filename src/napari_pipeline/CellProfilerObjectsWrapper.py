import numpy as np


class Objects:
    """Represents a segmentation of an image.

    IdentityPrimAutomatic produces three variants of its segmentation
    result. This object contains all three.

    There are three formats for segmentation, two of which support
    overlapping objects:

    get/set_segmented - Legacy, a single plane of labels that does not
                        support overlapping objects.

    (see cellprofiler_library.functions.segmentation._validate_labels)


    get_labels        - Supports overlapping objects, returns one or more planes
                        along with indices. A typical usage is to perform an
                        operation per-plane as if the objects did not overlap.

    (see cellprofiler_library.functions.segmentation.convert_dense_to_label_set)


    get/set_ijv       - Supports overlapping objects, returns a sparse
                        representation in which the first two columns are the
                        coordinates and the last is the object number. This
                        is efficient for doing things like calculating intensity
                        per-object.

    (see cellprofiler_library.functions.segmentation._validate_ijv)


    You can set one of the types and then get any of the types (except that
    get_segmented will raise an exception if objects overlap).
    """

    def __init__(self):
        self.__segmented = None
        self.__unedited_segmented = None
        self.__small_removed_segmented = None
        self.__parent_image = None

    @property
    def dimensions(self):
        if self.__parent_image:
            return self.__parent_image.dimensions

        shape = self.shape

        return len(shape)

    @property
    def volumetric(self):
        return self.dimensions == 3

    @property
    def masked(self):
        parent_image = self.parent_image

        assert parent_image is not None, "No parent image"

        return np.logical_and(self.segmented, parent_image.mask)

    @property
    def shape(self):
        dense, _ = self.__segmented.get_dense()

        if dense.shape[3] == 1:
            return dense.shape[-2:]

        return dense.shape[-3:]

    def get_segmented(self):
        """Get the de-facto segmentation of the image into objects: a matrix
        of object numbers.
        """
        return self.__segmentation_to_labels(self.__segmented)

    def set_segmented(self, labels):
        self.__segmented = self.__labels_to_segmentation(labels)

    segmented = property(get_segmented, set_segmented)

    @staticmethod
    def __labels_to_segmentation(labels):
        dense = convert_labels_to_dense(labels)
        return Segmentation(dense=dense)

    @staticmethod
    def __segmentation_to_labels(segmentation):
        assert isinstance(
            segmentation, Segmentation
        ), "Operation failed because objects were not initialized"

        dense, indices = segmentation.get_dense()

        assert (
            len(dense) == 1
        ), "Operation failed because objects overlapped. Please try with non-overlapping objects"

        if dense.shape[3] == 1:
            return dense.reshape(dense.shape[-2:])

        return dense.reshape(dense.shape[-3:])

    @property
    def indices(self):
        return indices_from_ijv(self.ijv, validate=False)

    @property
    def count(self):
        return count_from_ijv(self.ijv, validate=False)

    @property
    def areas(self):
        return areas_from_ijv(self.ijv, validate=False)

    def set_ijv(self, ijv, shape=None):
        """Set the segmentation to an IJV object format

        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        """
        sparse = convert_ijv_to_sparse(ijv)

        if shape is not None:
            shape = (1, 1, 1, shape[0], shape[1])

        self.__segmented = Segmentation(sparse=sparse, shape=shape)

    def get_ijv(self):
        """Get the segmentation in IJV object format

        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        """
        sparse = self.__segmented.sparse
        return convert_sparse_to_ijv(sparse, validate=False)

    ijv = property(get_ijv, set_ijv)

    def get_labels(self):
        """Get a set of labels matrices consisting of non-overlapping labels

        In IJV format, a single pixel might have multiple labels. If you
        want to use a labels matrix, you have an ambiguous situation and the
        resolution is to process separate labels matrices consisting of
        non-overlapping labels.

        returns a list of label matrixes and the indexes in each
        """
        dense, indices = self.__segmented.get_dense()

        return convert_dense_to_label_set(dense, indices=indices, validate=False)

    def has_unedited_segmented(self):
        """Return true if there is an unedited segmented matrix."""
        return self.__unedited_segmented is not None

    @property
    def unedited_segmented(self):
        """Get the segmentation of the image into objects, including junk that
        should be ignored: a matrix of object numbers.

        The default, if no unedited matrix is available, is the
        segmented labeling.
        """
        if self.__unedited_segmented is not None:
            return self.__segmentation_to_labels(self.__unedited_segmented)

        return self.segmented

    @unedited_segmented.setter
    def unedited_segmented(self, labels):
        self.__unedited_segmented = self.__labels_to_segmentation(labels)

    def has_small_removed_segmented(self):
        """Return true if there is a junk object matrix."""
        return self.__small_removed_segmented is not None

    @property
    def small_removed_segmented(self):
        """Get the matrix of segmented objects with the small objects removed

        This should be the same as the unedited_segmented label matrix with
        the small objects removed, but objects touching the sides of the image
        or the image mask still present.
        """
        if self.__small_removed_segmented is not None:
            return self.__segmentation_to_labels(self.__small_removed_segmented)

        return self.unedited_segmented

    @small_removed_segmented.setter
    def small_removed_segmented(self, labels):
        self.__small_removed_segmented = self.__labels_to_segmentation(labels)

    @property
    def parent_image(self):
        """The image that was analyzed to yield the objects.

        The image is an instance of CPImage which means it has the mask
        and crop mask.
        """
        return self.__parent_image

    @parent_image.setter
    def parent_image(self, parent_image):
        self.__parent_image = parent_image
        for segmentation in (
            self.__segmented,
            self.__small_removed_segmented,
            self.__unedited_segmented,
        ):
            if segmentation is not None and not segmentation.has_shape():
                shape = (
                    1,
                    1,
                    1,
                    parent_image.pixel_data.shape[0],
                    parent_image.pixel_data.shape[1],
                )
                segmentation.shape = shape

    @property
    def has_parent_image(self):
        """True if the objects were derived from a parent image

        """
        return self.__parent_image is not None

    def crop_image_similarly(self, image):
        """Crop an image in the same way as the parent image was cropped."""
        if image.shape == self.segmented.shape:
            return image
        if self.parent_image is None:
            raise ValueError("Images are of different size and no parent image")
        return self.parent_image.crop_image_similarly(image)

    def make_ijv_outlines(self, colors):
        """Make ijv-style color outlines

        Make outlines, coloring each object differently to distinguish between
        objects that might overlap.

        colors: a N x 3 color map to be used to color the outlines
        """
        return make_rgb_outlines(self.get_labels(), colors, validate=True)

    def relate_children(self, children):
        """Relate the object numbers in one label to the object numbers in another

        children - another "objects" instance: the labels of children within
                   the parent which is "self"

        Returns two 1-d arrays. The first gives the number of children within
        each parent. The second gives the mapping of each child to its parent's
        object number.
        """
        if self.volumetric:
            histogram = self.histogram_from_labels(self.segmented, children.segmented)
        else:
            histogram = self.histogram_from_ijv(self.ijv, children.ijv)

        return self.relate_histogram(histogram)

    def relate_labels(self, parent_labels, child_labels):
        """relate the object numbers in one label to those in another

        parent_labels - 2d label matrix of parent labels

        child_labels - 2d label matrix of child labels

        Returns two 1-d arrays. The first gives the number of children within
        each parent. The second gives the mapping of each child to its parent's
        object number.
        """
        histogram = self.histogram_from_labels(parent_labels, child_labels)
        return self.relate_histogram(histogram)

    @staticmethod
    def relate_histogram(histogram):
        """Return child counts and parents of children given a histogram

        histogram - histogram from histogram_from_ijv or histogram_from_labels
        """
        parent_count = histogram.shape[0] - 1

        parents_of_children = np.asarray(histogram.argmax(axis=0))
        if len(parents_of_children.shape) == 2:
            parents_of_children = np.squeeze(parents_of_children, axis=0)
        #
        # Create a histogram of # of children per parent
        children_per_parent = np.histogram(
            parents_of_children[1:], np.arange(parent_count + 2)
        )[0][1:]

        #
        # Make sure to remove the background elements at index 0
        #
        return children_per_parent, parents_of_children[1:]

    @staticmethod
    def histogram_from_labels(parent_labels, child_labels):
        """Find per pixel overlap of parent labels and child labels

        parent_labels - the parents which contain the children
        child_labels - the children to be mapped to a parent

        Returns a sparse matrix of overlap between each parent and child.
        Note that the first row and column are empty, as these
        correspond to parent and child labels of 0.
        """
        return find_label_overlaps(parent_labels, child_labels, validate=True)

    @staticmethod
    def histogram_from_ijv(parent_ijv, child_ijv):
        """Find per pixel overlap of parent labels and child labels,
        stored in ijv format.

        parent_ijv - the parents which contain the children
        child_ijv - the children to be mapped to a parent

        Returns a sparse matrix of overlap between each parent and child.
        Note that the first row and column are empty, as these
        correspond to parent and child labels of 0.
        """
        return find_ijv_overlaps(parent_ijv, child_ijv, validate=True)

    def fn_of_label_and_index(self, func):
        """Call a function taking a label matrix with the segmented labels

        function - should have signature like
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        """
        return func(self.segmented, self.indices)

    def fn_of_ones_label_and_index(self, func):
        """Call a function taking an image, a label matrix and an index with an image of all ones

        function - should have signature like
                   image  - image with same dimensions as labels
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        Pass this function an "image" of all ones, for instance to compute
        a center or an area
        """
        return func(np.ones(self.segmented.shape), self.segmented, self.indices)

    def center_of_mass(self):
        return center_of_labels_mass(self.segmented, validate=False)

    def overlapping(self):
        if not isinstance(self.__segmented, Segmentation):
            return False
        dense, indices = self.__segmented.get_dense()
        return len(dense) != 1
class Segmentation:
    """A segmentation of a space into labeled objects

    Supports overlapping objects and cacheing. Retrieval can be as a
    single plane (legacy), as multiple planes and as sparse ijv.
    """

    def __init__(self, dense=None, sparse=None, shape=None):
        """Initialize the segmentation with either a dense or sparse labeling

        dense - A 6-d array composed of one or more 5-d integer labelings of
        each hyper-voxel. The dimension order is labeling, c, t, z, y, x.
        Typically, a 2-D non-overlapping segmentation has dimensions of
        1, 1, 1, 1, y, x.

        sparse - a labeling stored in a record data type with each column
                 having a name of "c", "t", "z", "y", "x" or "label".
                 The "label" column is the object number, starting with 1.
                 When "c", "t", "z" are absent, this is the interoperable with
                 an ijv labeling of the pixels, or in the COOrdinate format
                 (as in scipy.sparse.coo_matrix).

        shape - the 5-D shape of the imaging site if sparse. If this is absent
                the shape is inferred from the given coordinates of the sparse
                labeling, which is a size capable of containing an equivilent
                dense representation, but may not be exactly equal to the
                original shape of the imaging site.
        """
        if dense is not None:
            _validate_dense(dense)
        if sparse is not None:
            _validate_sparse(sparse)
        if shape is not None:
            _validate_dense_shape(shape)

        self.__dense = dense
        self.__sparse = sparse
        if shape is not None:
            self.__shape = shape
            self.__explicit_shape = True
        else:
            self.__shape = None
            self.__explicit_shape = False

        if dense is not None:
            self.__indices = indices_from_dense(dense)

    @property
    def shape(self):
        """Get or estimate the shape of the segmentation matrix
        This is the dense shape, ('c', 't', 'z', 'y', 'x')

        Order of precedence:
        Shape supplied in the constructor
        Shape of the dense representation
        maximum extent of the sparse representation + 1
        """
        if self.__shape is not None:
            return self.__shape
        if self.has_dense():
            self.__shape = self.get_dense()[0].shape[1:]
        else:
            sparse = self.sparse
            if len(sparse) == 0:
                self.__shape = (1, 1, 1, 1, 1)
            else:
                self.__shape = dense_shape_from_sparse(sparse, validate=False)
        return self.__shape

    @shape.setter
    def shape(self, shape):
        """Set the shape of the segmentation array

        shape - the 5D shape of the array

        This fixes the shape of the 5D array for sparse representations
        """
        self.__shape = shape
        self.__explicit_shape = True

    def has_dense(self):
        return self.__dense is not None

    def has_sparse(self):
        return self.__sparse is not None

    def has_shape(self):
        if self.__explicit_shape:
            return True

        return self.has_dense()

    @property
    def sparse(self):
        """Get the sparse representation of the segmentation

        returns a Numpy record array where every row represents
        the labeling of a pixel. The dtype record names are taken from
        HDF5ObjectSet.AXIS_[C,T,Z,Y,X] and AXIS_LABELS for the object
        numbers.
        """
        if self.__sparse is not None:
            return self.__sparse

        if not self.has_dense():
            raise ValueError("Can't find object dense segmentation.")

        return self.__convert_dense_to_sparse()

    def get_dense(self):
        """Get the dense representation of the segmentation

        return the segmentation as a 6-D array and a sequence of arrays of the
        object numbers in each 5-D hyperplane of the segmentation. The first
        axis of the segmentation allows us to assign multiple labels to
        individual pixels. Given a 5-D algorithm, the code typically iterates
        over the first axis:

        for labels in self.get_dense():
            # do something

        The remaining axes are in the order, c, t, z, y and x
        """
        if self.__dense is not None:
            return self.__dense, self.__indices

        if not self.has_sparse():
            raise ValueError("Can't find object sparse segmentation.")

        return self.__convert_sparse_to_dense()

    def __convert_dense_to_sparse(self):
        dense, _ = self.get_dense()
        sparse = convert_dense_to_sparse(dense, validate=False)
        self.__sparse = sparse
        return sparse

    def __set_dense(self, dense, indices=None):
        self.__dense = dense

        if indices is not None:
            self.__indices = indices
        else:
            self.__indices = indices_from_dense(dense)

        return dense, self.__indices

    def __convert_sparse_to_dense(self):
        dense, indices = convert_sparse_to_dense(
            self.sparse, self.shape, validate=False)

        return self.__set_dense(dense, indices)
# Segmentation related functions
from enum import Enum
from numpy.random.mtrand import RandomState
import scipy.sparse
import centrosome.index

class SPARSE_FIELD(Enum):
    label = "label"
    c = "c"
    t = "t"
    z = "z"
    y = "y"
    x = "x"
    
class DENSE_AXIS(Enum):
    label_idx = 0
    c = 1
    t = 2
    z = 3
    y = 4
    x = 5

SPARSE_FIELDS = tuple([mem.value for mem in SPARSE_FIELD])
SPARSE_AXES_FIELDS = SPARSE_FIELDS[1:]
DENSE_AXIS_NAMES = tuple([mem.name for mem in DENSE_AXIS])
DENSE_SHAPE_NAMES = DENSE_AXIS_NAMES[1:]

# ------ Functions for validating segmentation formats ------

def _validate_dense(dense):
    """
    A 'dense' matrix is a 6 dimensional array with axis order:
    (label_idx, c, t, z, y, x)

    When the 'label_idx' dim = 1, it hosts zero or more non-overlapping labels
    When the 'label_idx' dim > 1, each index hosts one or more non-overlapping
    labels (within that index)
    In other words, while labels within an index of 'label_idx' are never
    overlapping, labels between indices of 'label_idx' would overlap
    i.e. 'dense.sum(axis=0)' is invalid, producing innaccurate labels

    A 'dense' matrix is usually paired with an array of indices specifying
    which label values are present in which index of the 'label_idx' dim
    (see 'indices_from_dense' for more details)
    """
    ndim = len(DENSE_AXIS_NAMES)
    assert type(dense) == np.ndarray, "dense must be ndarray"
    assert dense.ndim == ndim, \
    f"dense must be {ndim}-dimensional - f{DENSE_AXIS_NAMES}"

def _validate_dense_shape(dense_shape):
    """
    'dense_shape', as opposed to 'dense.shape', is the shape of the 'dense'
    matrix sans the 'label_idx' axis, i.e.
    (c, t, z, y, z)
    """
    ndim = len(DENSE_SHAPE_NAMES)
    assert (dense_shape is None or
            len(dense_shape) == ndim
    ), f"dense_shape must be length {ndim}, omitting '{DENSE_AXIS.label_idx.name}' dim"

def _validate_labels(labels):
    """
    A 'labels' matrix is another, more constrained, dense representation

    It is strictly 2- or 3-dimensional, of shape: (y, x) or (z, y, x)
    A single 'labels' matrix does not allow for overlapping labels within it

    It is essentially a 'dense' of shape (1, 1, 1, 1, y, x), but squeezed
    such that the ('label_idx', 'c', 't', 'z') axes are removed

    For a 'dense' with shape (2+, 1, 1, 1, y, x), a 'label_set' can be
    constructed (see 'convert_dense_to_label_set' for more details)
    """
    assert type(labels) == np.ndarray, "labels must be ndarray"
    assert (
        labels.ndim == 2 or
        labels.ndim == 3
    ), "labels must be 2- or 3-dimensional"

def _validate_sparse(sparse):
    """
    'sparse' is a sparse representation of labelings
    It's either a numpy recarray, or castable as such via
    'arr.view(np.recarray)'
    where the data types are typed fields who's names are a subset of:
    set('label', 'c', 't', 'z', 'y', 'x')
    and where the data is a 1-dimensional array of tuples, matching the fields

    e.g.
    rec.array([(0, 0, 0, 1), (0, 1, 0, 1), (1, 0, 0, 1), (1, 1, 0, 1),
               (0, 1, 0, 2), (0, 1, 1, 2), (1, 1, 0, 2), (1, 1, 1, 2)],
              dtype=[('z', '<u2'), ('y', '<u2'), ('x', '<u2'), ('label', 'u1')])

    Note that each field may have its own type, the tuple order matches the
    field order, each tuple is unique by at least one element, and there is
    no need (although permissable) to specify a field if it will be the same
    among all the data values (i.e. 'c' and 't' not specified above as they
    would all be '0')

    Note also there is no tuple who's 'label' value is '0' because including
    all '0' values would make 'sparse' equivilent in memory to 'dense',
    and including any '0' values at all would be including a non-label
    (i.e. background)
    """
    assert (
        type(sparse) == np.ndarray or
        type(sparse) == np.recarray
    ), "sparse must be ndarray or recarray"

    assert sparse.ndim == 1, "sparse mut be 1-dimensional"

    axes = sparse.dtype.names

    assert axes is not None, "sparse must have dtype fields"

    axes_set = set(axes)
    full_set = set(SPARSE_FIELDS)

    assert len(axes) == len(axes_set), "duplicate dtype fields in sparse"
    assert axes_set.issubset(full_set), "sparse has unknown dtype fields"

def _validate_ijv(ijv):
    """
    'ijv' is another, more constrained, sparse representation

    It is a 2-dimensional array of triplets, of shape: (num_coords, 3)
    It also allows for overlapping labels

    Unlike 'sparse', it is strictly an ndarray with a single dtype
    for all values, and is strictly in triplet form, where
    [i, j, v] = [y_coord, x_coord, label_value]
    It cannot host values for 'c', 't' or 'z'

    e.g. this 'sparse' record:
    rec.array([(0, 0, 0, 1), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 1, 2),],
             dtype=[('z', '<u2'), ('y', '<u2'), ('x', '<u2'), ('label', 'u1')])

    would equate to this 'ijv' matrix:
    array([[0, 0, 1],
           [1, 0, 1],
           [1, 0, 2],
           [1, 1, 2]],
          dtype=uint16)
    """
    assert type(ijv) == np.ndarray, "ijv must be ndarray"
    assert ijv.ndim == 2, "ijv must be 2-dimensional"
    assert ijv.shape[1] == 3, "ijv must have 3 columns"

# ------ Functions converting between segmentation formats ------

def indices_from_dense(dense, validate=True):
    """
    Retrieve indices of a 'dense' matrix

    The return value is a python list of 1-dimensional ndarrays
    whose lengths may or may not be equal (non-homogeneous)

    A given index of the 'indices' list corresponds to the same index in the
    'label_idx' axis of the 'dense' matrix
    An element of the 'indices' list is a 1-dimensional ndarray of labels that
    are present in the 'dense' matrix, at that index

    e.g. the following 'indices' list:
    [array([1, 2, 4], dtype=uint8), array([3, 5], dtype=uint8)]

    specifies that its corresponding 'dense' matrix has a 'label_idx' dim = 2,
    i.e. a shape of: (2, c, t, z, y, x),
    where the labels '1', '2', '4' are present at index '0', and the labels '3'
    and '4' are present at index '1'
    """
    if validate:
        _validate_dense(dense)

    indices = [np.unique(d) for d in dense]
    indices = [idx[1:] if idx[0] == 0 else idx for idx in indices]
    return indices

def dense_shape_from_sparse(sparse, validate=True):
    if validate:
        _validate_sparse(sparse)

    return tuple([
        np.max(sparse[axis]) + 2
        if axis in list(sparse.dtype.fields.keys())
        else 1
        for axis in SPARSE_AXES_FIELDS
    ])

def indices_from_ijv(ijv, validate=True):
    """
    Get the indices for a scipy.ndimage-style function from the 'ijv' formatted
    segmented labels
    """
    if validate:
        _validate_ijv(ijv)

    if len(ijv) == 0:
        return np.zeros(0, np.int32)

    max_label = np.max(ijv[:, 2])

    return np.arange(max_label).astype(np.int32) + 1

def count_from_ijv(ijv, indices=None, validate=True):
    """
    The number of labels in an 'ijv' formatted label matrix
    """
    if validate:
        _validate_ijv(ijv)

    if indices is None:
        indices = indices_from_ijv(ijv, validate=False)

    return len(indices)

def areas_from_ijv(ijv, indices=None, validate=True):
    """
    The area of each object in an 'ijv' formatted label matrix

    Because of the discrete nature of the matrix, this is simply equal to the
    occurrence count of each label in the matrix
    """
    if validate:
        _validate_ijv(ijv)

    if indices is None:
        indices = indices_from_ijv(ijv, validate=False)

    if len(indices) == 0:
        return np.zeros(0, int)

    return np.bincount(ijv[:, 2])[indices]

def downsample_labels(labels, validate=True):
    """
    Convert a 'labels' matrix to the smallest possible integer format
    """
    if validate:
        _validate_labels(labels)

    labels_max = np.max(labels)
    if labels_max < 128:
        return labels.astype(np.int8)
    elif labels_max < 32768:
        return labels.astype(np.int16)
    return labels.astype(np.int32)

def convert_dense_to_label_set(dense, indices=None, validate=True):
    """
    Convert a 'dense' matrix into a list of 2-tuples,
    where the number of tuples corresponds to the 'label_idx' dim of the
    'dense' matrix (see '_validate_dense' for details),
    the tuple's first element is a 'labels' matrix
    (see '_validate_labels' for details),
    and the tuple's second element is the 1-d ndarray of labels in the matrix
    (see 'indices_from_dense' for details)
    """
    if validate:
        _validate_dense(dense)
        assert(
            dense.shape[DENSE_AXIS.c.value] == 1 and 
            dense.shape[DENSE_AXIS.t.value] == 1
        ), f"dense must have shape where f{DENSE_AXIS.c.name} = 1 and f{DENSE_AXIS.t.name} = 1"

    if indices is None:
        indices = indices_from_dense(dense, validate=False)

    label_set_len = dense.shape[DENSE_AXIS.label_idx.value]
    squeezed_dense = dense.squeeze()

    if label_set_len == 1:
        return [(squeezed_dense, indices[0])]
    
    return [(squeezed_dense[i], indices[i]) for i in range(label_set_len)]

def indices_from_labels(labels, validate=True):
    if validate:
        _validate_labels(labels)

    return np.unique(labels[labels != 0])

def cast_labels_to_label_set(labels, validate=True):
    """
    Takes in a 'labels' matrix and casts it into a 1-element 'label_set'
    """
    if validate:
        _validate_labels(labels)

    return [(labels, indices_from_labels(labels, validate=False))]

def convert_labels_to_dense(labels, validate=True):
    """
    Convert a 'labels' matrix (e.g. scipy.ndimage.label) to 'dense' matrix
    """
    if validate:
        _validate_labels(labels)

    typed_labels = downsample_labels(labels, validate=False)

    if labels.ndim == 3:
        expand_axes = (
            DENSE_AXIS.label_idx.value,
            DENSE_AXIS.c.value,
            DENSE_AXIS.t.value
        )
    else:
        expand_axes = (
            DENSE_AXIS.label_idx.value,
            DENSE_AXIS.c.value,
            DENSE_AXIS.t.value,
            DENSE_AXIS.z.value
        )

    return np.expand_dims(typed_labels, axis=expand_axes)

def convert_dense_to_sparse(dense, validate=True):
    if validate:
        _validate_dense(dense)

    full_shape = dense.shape
    label_dim = full_shape[DENSE_AXIS.label_idx.value]
    dense_shape = tuple(
        [full_shape[DENSE_AXIS[n].value] for n in DENSE_SHAPE_NAMES]
    )

    axes_labels = np.array(SPARSE_AXES_FIELDS)
    axes = axes_labels[np.where(np.array(dense_shape) > 1)]

    compact = np.squeeze(dense)
    if label_dim == 1:
        compact = np.expand_dims(compact, axis=0)

    coords = np.where(compact != 0)
    labels = compact[coords]
    # no longer need the labels dim
    coords = coords[1:]

    if np.max(compact.shape) < 2 ** 16:
        coords_dtype = np.uint16
    else:
        coords_dtype = np.uint32

    if len(labels) > 0:
        max_label = np.max(labels)
        if max_label < 2 ** 8:
            labels_dtype = np.uint8
        elif max_label < 2 ** 16:
            labels_dtype = np.uint16
        else:
            labels_dtype = np.uint32
    else:
        labels_dtype = np.uint8

    dtype = [(axis, coords_dtype) for axis in axes]
    dtype.append((SPARSE_FIELD.label.value, labels_dtype))
    sparse = np.core.records.fromarrays(list(coords) + [labels], dtype=dtype)

    return sparse

def convert_ijv_to_sparse(ijv, validate=True):
    if validate:
        _validate_ijv(ijv)

    return np.core.records.fromarrays(
        (ijv[:, 0], ijv[:, 1], ijv[:, 2]),
        [
            (SPARSE_FIELD.y.value, ijv.dtype),
            (SPARSE_FIELD.x.value, ijv.dtype),
            (SPARSE_FIELD.label.value, ijv.dtype)
        ],
    )

def convert_sparse_to_ijv(sparse, validate=True):
    if validate:
        _validate_sparse(sparse)

    return np.column_stack([sparse[axis] for axis in (
        SPARSE_FIELD.y.value, SPARSE_FIELD.x.value, SPARSE_FIELD.label.value)
    ])

def convert_labels_to_ijv(labels, validate=True):
    if validate:
        _validate_labels(labels)

    dense = convert_labels_to_dense(labels, validate=False)
    sparse = convert_dense_to_sparse(dense, validate=False)
    ijv = convert_sparse_to_ijv(sparse, validate=False)

    return ijv

def convert_ijv_to_label_set(ijv, dense_shape=None, validate=True):
    if validate:
        _validate_ijv(ijv)

    sparse = convert_ijv_to_sparse(ijv, validate=False)

    if dense_shape is None:
        dense_shape = dense_shape_from_sparse(sparse)

    dense, indices = convert_sparse_to_dense(
        sparse,
        dense_shape=dense_shape,
        validate=False
    )

    label_set = convert_dense_to_label_set(
        dense,
        indices=indices,
        validate=False
    )

    return label_set

def convert_label_set_to_ijv(label_set, validate=True):
    return np.concatenate(
        [convert_labels_to_ijv(l[0], validate) for l in label_set],
        axis=0
    )

def convert_sparse_to_dense(sparse, dense_shape=None, validate=True):
    """
    Convert 'sparse' representation to 'dense' matrix

    Returns 'dense' matrix and corresponding 'indices'
    """
    if validate:
        _validate_sparse(sparse)
        _validate_dense_shape(dense_shape)

    if len(sparse) == 0:
        if dense_shape is None:
            dense_shape = tuple([1 for _ in range(len(DENSE_SHAPE_NAMES))])

        dense = np.expand_dims(
            np.zeros(dense_shape, np.uint8),
            axis=DENSE_AXIS.label_idx.value
        )

        return dense, indices_from_dense(dense, validate=False)

    if dense_shape is None:
        dense_shape = dense_shape_from_sparse(sparse, validate=False)

    #
    # The code below assigns a "color" to each label so that no
    # two labels have the same color
    #
    positional_columns = []
    available_columns = []
    lexsort_columns = []
    for axis in SPARSE_AXES_FIELDS:
        if axis in list(sparse.dtype.fields.keys()):
            positional_columns.append(sparse[axis])
            available_columns.append(sparse[axis])
            lexsort_columns.insert(0, sparse[axis])
        else:
            positional_columns.append(0)
    labels = sparse[SPARSE_FIELD.label.value]
    lexsort_columns.insert(0, labels)

    sort_order = np.lexsort(lexsort_columns)
    n_labels = np.max(labels)
    #
    # Find the first of a run that's different from the rest
    #
    mask = (
        available_columns[0][sort_order[:-1]]
        != available_columns[0][sort_order[1:]]
    )
    for column in available_columns[1:]:
        mask = mask | (column[sort_order[:-1]] != column[sort_order[1:]])
    breaks = np.hstack(([0], np.where(mask)[0] + 1, [len(labels)]))
    firsts = breaks[:-1]
    counts = breaks[1:] - firsts
    #
    # Eliminate the locations that are singly labeled
    #
    mask = counts > 1
    firsts = firsts[mask]
    counts = counts[mask]
    if len(counts) == 0:
        dense = np.zeros([1] + list(dense_shape), labels.dtype)
        dense[tuple([0] + positional_columns)] = labels
        return dense, indices_from_dense(dense, validate=False)
    #
    # There are n * n-1 pairs for each coordinate (n = # labels)
    # n = 1 -> 0 pairs, n = 2 -> 2 pairs, n = 3 -> 6 pairs
    #
    pairs = centrosome.index.all_pairs(np.max(counts))
    pair_counts = counts * (counts - 1)
    #
    # Create an indexer for the inputs (indexes) and for the outputs
    # (first and second of the pairs)
    #
    # Remember idx points into sort_order which points into labels
    # to get the nth label, grouped into consecutive positions.
    #
    output_indexer = centrosome.index.Indexes(pair_counts)
    #
    # The start of the run of overlaps and the offsets
    #
    run_starts = firsts[output_indexer.rev_idx]
    offs = pairs[output_indexer.idx[0], :]
    first = labels[sort_order[run_starts + offs[:, 0]]]
    second = labels[sort_order[run_starts + offs[:, 1]]]
    #
    # And sort these so that we get consecutive lists for each
    #
    pair_sort_order = np.lexsort((second, first))
    #
    # Eliminate dupes
    #
    to_keep = np.hstack(
        ([True], (first[1:] != first[:-1]) | (second[1:] != second[:-1]))
    )
    to_keep = to_keep & (first != second)
    pair_idx = pair_sort_order[to_keep]
    first = first[pair_idx]
    second = second[pair_idx]
    #
    # Bincount each label so we can find the ones that have the
    # most overlap. See cpmorphology.color_labels and
    # Welsh, "An upper bound for the chromatic number of a graph and
    # its application to timetabling problems", The Computer Journal, 10(1)
    # p 85 (1967)
    #
    overlap_counts = np.bincount(first.astype(np.int32))
    #
    # The index to the i'th label's stuff
    #
    indexes = np.cumsum(overlap_counts) - overlap_counts
    #
    # A vector of a current color per label. All non-overlapping
    # objects are assigned to plane 1
    #
    v_color = np.ones(n_labels + 1, int)
    v_color[0] = 0
    #
    # Clear all overlapping objects
    #
    v_color[np.unique(first)] = 0
    #
    # The processing order is from most overlapping to least
    #
    ol_labels = np.where(overlap_counts > 0)[0]
    processing_order = np.lexsort((ol_labels, overlap_counts[ol_labels]))

    for index in ol_labels[processing_order]:
        neighbors = second[indexes[index] : indexes[index] + overlap_counts[index]]
        colors = np.unique(v_color[neighbors])
        if colors[0] == 0:
            if len(colors) == 1:
                # all unassigned - put self in group 1
                v_color[index] = 1
                continue
            else:
                # otherwise, ignore the unprocessed group and continue
                colors = colors[1:]
        # Match a range against the colors array - the first place
        # they don't match is the first color we can use
        crange = np.arange(1, len(colors) + 1)
        misses = crange[colors != crange]
        if len(misses):
            color = misses[0]
        else:
            max_color = len(colors) + 1
            color = max_color
        v_color[index] = color
    #
    # Create the dense matrix by using the color to address the
    # 5-d hyperplane into which we place each label
    #
    dense = np.zeros([np.max(v_color)] + list(dense_shape), labels.dtype)
    slices = tuple([v_color[labels] - 1] + positional_columns)
    dense[slices] = labels
    indices = [np.where(v_color == i)[0] for i in range(1, dense.shape[0] + 1)]

    return dense, indices

# ------ Functions for operating on segmentation formats ------

def make_rgb_outlines(label_set, colors, random_seed=None, validate=True):
    """
    Assign rgb colors to outlines of labels in 'label_set`

    Make outlines, coloring each object differently to distinguish between
    objects that might overlap.

    'label_set': see 'convert_dense_to_label_set'

    'colors': a N x 3 color map to be used to color the outlines
    where N in dim 0 should match the number of unique labels in the
    `label_set`, and values are R, G, and B values normalized to [0, 1]

    'random_seed' when provided, will seed the RNG for permuting colors
    between 'labels' matrices in the 'label_set'
    """
    if validate:
        assert type(colors) == np.ndarray, "'colors' must be ndarray"
        assert (
            colors.ndim == 2 and
            colors.shape[1] == 3
        ), "'colors' must be of shape (N, 3)"
        indices = [i for _, idxs in label_set for i in idxs]
        # >= because technically you can have superflous colors (but don't)
        assert colors.shape[0] >= len(indices), \
            "axis 1 of 'colors' must be equal to the number of unique labels in 'label_set'"
    #
    # Get planes of non-overlapping objects. The idea here is to use
    # the most similar colors in the color space for objects that
    # don't overlap.
    #
    label_outline_set = [
        (centrosome.outline.outline(label), indexes)
        for label, indexes in label_set
    ]
    rgb_image = np.zeros(list(label_outline_set[0][0].shape) + [3], np.float32)
    #
    # Find out how many unique labels in each
    #
    counts = [np.sum(np.unique(l) != 0) for l, _ in label_outline_set]
    if len(counts) == 1 and counts[0] == 0:
        return rgb_image

    if len(colors) < len(label_outline_set):
        # Have to color 2 planes using the same color!
        # There's some chance that overlapping objects will get
        # the same color. Give me more colors to work with please.
        colors = np.vstack([colors] * (1 + len(label_outline_set) // len(colors)))
    r = RandomState()
    r.seed(random_seed)
    alpha = np.zeros(label_outline_set[0][0].shape, np.float32)
    order = np.lexsort([counts])

    for idx, i in enumerate(order):
        max_available = len(colors) / (len(label_outline_set) - idx)
        ncolors = min(counts[i], max_available)
        my_colors = colors[:ncolors]
        colors = colors[ncolors:]
        my_colors = my_colors[r.permutation(np.arange(ncolors))]
        my_labels, indexes = label_outline_set[i]
        color_idx = np.zeros(np.max(indexes) + 1, int)
        color_idx[indexes] = np.arange(len(indexes)) % ncolors
        rgb_image[my_labels != 0, :] += my_colors[
            color_idx[my_labels[my_labels != 0]], :
        ]
        alpha[my_labels != 0] += 1
    rgb_image[alpha > 0, :] /= alpha[alpha > 0][:, np.newaxis]

    return rgb_image

# needs library tests
def find_label_overlaps(parent_labels, child_labels, validate=True):
    """
    Find per pixel overlap of parent labels and child labels

    'parent_labels' - the parents which contain the children in 'labels' format
    'child_labels' - the children to be mapped to a parent in 'labels' format

    Returns a sparse 'coo_matrix' of overlap between each parent and child.
    Note that the first row and column are empty, as these
    correspond to parent and child labels of 0.
    """
    if validate:
        _validate_labels(parent_labels)
        _validate_labels(child_labels)

    parent_count = np.max(parent_labels)
    child_count = np.max(child_labels)
    #
    # If the labels are different shapes, crop to shared shape.
    #
    common_shape = np.minimum(parent_labels.shape, child_labels.shape)

    if parent_labels.ndim == 3:
        parent_labels = parent_labels[
            0 : common_shape[0], 0 : common_shape[1], 0 : common_shape[2]
        ]
        child_labels = child_labels[
            0 : common_shape[0], 0 : common_shape[1], 0 : common_shape[2]
        ]
    else:
        parent_labels = parent_labels[0 : common_shape[0], 0 : common_shape[1]]
        child_labels = child_labels[0 : common_shape[0], 0 : common_shape[1]]

    #
    # Only look at points that are labeled in parent and child
    #
    not_zero = (parent_labels > 0) & (child_labels > 0)
    not_zero_count = np.sum(not_zero)

    #
    # each row (axis = 0) is a parent
    # each column (axis = 1) is a child
    #
    return scipy.sparse.coo_matrix(
        (
            np.ones((not_zero_count,)),
            (parent_labels[not_zero], child_labels[not_zero]),
        ),
        shape=(parent_count + 1, child_count + 1),
    )

# needs library tests
def find_ijv_overlaps(parent_ijv, child_ijv, validate=True):
    """
    Find per pixel overlap of parent labels and child labels

    'parent_ijv' - the parents which contain the children, in 'ijv' format
    'child_ijv' - the children to be mapped to a parent, in 'ijv' format

    Returns a sparse 'csc_matrix' of overlap between each parent and child.
    Note that the first row and column are empty, as these
    correspond to parent and child labels of 0.
    """
    if validate:
        _validate_ijv(parent_ijv)
        _validate_ijv(child_ijv)

    parent_count = 0 if (parent_ijv.shape[0] == 0) else np.max(parent_ijv[:, 2])
    child_count = 0 if (child_ijv.shape[0] == 0) else np.max(child_ijv[:, 2])

    if parent_count == 0 or child_count == 0:
        return np.zeros((parent_count + 1, child_count + 1), int)

    dim_i = max(np.max(parent_ijv[:, 0]), np.max(child_ijv[:, 0])) + 1
    dim_j = max(np.max(parent_ijv[:, 1]), np.max(child_ijv[:, 1])) + 1
    parent_linear_ij = parent_ijv[:, 0] + dim_i * parent_ijv[:, 1].astype(
        np.uint64
    )
    child_linear_ij = child_ijv[:, 0] + dim_i * child_ijv[:, 1].astype(np.uint64)

    parent_matrix = scipy.sparse.coo_matrix(
        (np.ones((parent_ijv.shape[0],)), (parent_ijv[:, 2], parent_linear_ij)),
        shape=(parent_count + 1, dim_i * dim_j),
    )
    child_matrix = scipy.sparse.coo_matrix(
        (np.ones((child_ijv.shape[0],)), (child_linear_ij, child_ijv[:, 2])),
        shape=(dim_i * dim_j, child_count + 1),
    )
    # I surely do not understand the sparse code.  Converting both
    # arrays to csc gives the best peformance... Why not p.csr and
    # c.csc?
    return parent_matrix.tocsc() * child_matrix.tocsc()

def center_of_labels_mass(labels, validate=True):
    if validate:
        _validate_labels(labels)

    indices = indices_from_labels(labels)
    return np.array(
        scipy.ndimage.center_of_mass(np.ones_like(labels), labels, indices)
    )
