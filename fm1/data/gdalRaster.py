import numpy as np, os, cv2

from osgeo import gdal, osr

class GdalRaster():
    '''
    Thin wrapper around gdal.Dataset
    '''
    def __init__(self, path, gray=False, resample='bilinear'):
        super().__init__()
        self.path = path

        self.dset = gdal.Open(path, gdal.GA_ReadOnly)
        self.srs = self.dset.GetProjectionRef()
        self.nbands = self.dset.RasterCount
        assert self.nbands == 1 or self.nbands == 3 or self.nbands == 4, 'Only Grayscale/RGB/RGB-NIR tiffs (1, 3, or 4 channels) are supported'

        self.gray = False

        self.w = self.dset.RasterXSize
        self.h = self.dset.RasterYSize

        self.resample = gdal.GRIORA_Bilinear
        if resample == 'nearest':
            self.resample = gdal.GRIORA_NearestNeighbour

        gt = self.dset.GetGeoTransform()
        self.prj_from_pix = np.array((
            gt[1], gt[2], gt[0],
            gt[4], gt[5], gt[3]), dtype=np.float64).reshape(2,3)
        self.pix_from_prj = np.linalg.inv(np.vstack((self.prj_from_pix, np.array((0,0,1.)))))[:2]

        self.gsdAtCenter = self._computeGsd()

        self.wmTlbr = self._computeWmTlbr()
        self.defaultResample = 'bilinear'

    def readPixelRange(self, pixelTlwh, w, h, c):
        # data = self.dset.ReadAsArray(pixelTlwh[0], pixelTlwh[1], pixelTlwh[2], pixelTlwh[3], resample_alg='bilinear', buf_xsize=w, buf_ysize=h).transpose(1,2,0)
        data = self.dset.ReadAsArray(pixelTlwh[0], pixelTlwh[1], pixelTlwh[2], pixelTlwh[3], resample_alg=self.resample, buf_xsize=w, buf_ysize=h)
        if data is None: return None
        data = data.transpose(1,2,0)

        if data.shape[-1] == 4: data = data[...,:3]
        if c == 1 and data.shape[-1] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        if c == 3 and data.shape[-1] == 1:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        if data.ndim == 2: data = data[...,None]

        return data

    def _computeGsd(self):
        '''
        Compute the "ground sample distance", the number of meters in between the centers of two pixels.
        Note: Unlike WM meters, which are scaled according to latitude, real euclidean/cartesian/ecef meters are used.

        Note: If you use the result of this function, be warned it is assuming the scale factor of whatever
              projection used is closely constant to that at the center pixel.
        '''

        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(self.dset.GetProjection())

        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4978)

        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER);
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER);
        transform = osr.CoordinateTransformation(src_srs, dst_srs)


        pixPt1 = np.array((self.w / 2, self.h / 2, 0), dtype=np.float64)
        pixPt2 = np.array((1 + self.w / 2, 1 + self.h / 2, 0), dtype=np.float64)

        prjPt1 = np.array((*(self.prj_from_pix @ pixPt1), 0.))
        prjPt2 = np.array((*(self.prj_from_pix @ pixPt2), 0.))

        ecefPt1 = np.array(transform.TransformPoints(prjPt1[None])[0])
        ecefPt2 = np.array(transform.TransformPoints(prjPt2[None])[0])

        pixPtDist = np.linalg.norm(pixPt2 - pixPt1)
        ecefPtDist = np.linalg.norm(ecefPt2 - ecefPt1)
        # print('pixPtDist :', pixPtDist)
        # print('ecefPtDist:', ecefPtDist)

        gsd = ecefPtDist / pixPtDist
        # print(self.path, f'gsdAtCenter: {gsd:>5.2f}m')

        return gsd

    def _computeWmTlbr(self):
        pixPts = np.array((
            0, 0, 1,
            self.w, 0, 1,
            self.w, self.h, 1,
            0, self.h, 1.)).reshape(-1,3)
        prjPts = np.zeros_like(pixPts)
        prjPts[:,:2] = pixPts @ self.prj_from_pix.T

        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(self.dset.GetProjection())

        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(3857)

        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER);
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER);
        transform = osr.CoordinateTransformation(src_srs, dst_srs)

        wmPts = np.array(transform.TransformPoints(prjPts))

        tl = wmPts[:,:2].min(0)
        br = wmPts[:,:2].max(0)
        return np.array((*tl, *br))
