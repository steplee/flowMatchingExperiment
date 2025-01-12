import numpy as np, os, cv2

from osgeo import gdal, osr
# from ..earth import Earth

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





    def readWmRange(self, wmTlbr, w, h, resample=None, errorThreshold=None):
        '''
        Return an image of size `w` and `h` filled with the dataset's values from the WM bounding box `wmTlbr`.
        The underlying dataset needn't be WM projected, thanks to GDAL's warp functionality.
        But this function probably executes faster and with higher quality if it were.

        NOTE: This function is slow: see the `readWmRangeFaster` function below if sampling millions of chips.
        '''

        # TODO: If the dataset is wm, use a faster RasterIO/ReadAsArray (rather than warp) based approach

        resample = resample or self.defaultResample

        warp_option_dict = {
                'format': 'MEM',
                'outputBounds': wmTlbr,
                'outputBoundsSRS': 'EPSG:3857',
                'dstSRS': 'EPSG:3857',
                'width': w,
                'height': h,
                'resampleAlg': resample,
                'dstNodata': 0,
                # 'multithread': True, # 5% slower?
                # 'errorThreshold': 50, # no discernable speedup
                #'srcBands': [1,2,3] if self.nbands == 3 else [1], # WARNING: Not supported in my version of GDAL (but latest online docs show it...)

                'errorThreshold': errorThreshold
                }

        temp_dataset = gdal.Warp(destNameOrDestDS='', srcDSOrSrcDSTab=self.dset, options=gdal.WarpOptions(**warp_option_dict))

        if not temp_dataset:
            raise RuntimeError('gdal warp failed')

        data = temp_dataset.ReadAsArray()
        if data.ndim == 2: data = data[None]
        data = data.transpose(1,2,0)
        del temp_dataset

        return data

    def readWmRangeFaster(self, wmTlbr, w, h, c, resample=None, oversample=1):
        '''
        The GDAL Warp function is very slow.
        I believe it is the bottleneck for indexing (when > hundreds of thousands of chips are needed, it becomes an issue).

        So here is a warp function that **approximates warping from any projection and any GSD to some query WM box**.
        It does this by using **only four** point samples that are actually warped.
        These four samples are at the corners of the input query image. A homography will related the _sampled image_ in tiff-native/"projected" coordinates, to the _output image_ in WM.

        The GDAL Warp options seem like they ought to offer a similar tradeoff with the `errorThreshold` parameter, but in my tests it makes
        no difference.

        This function __ought__ to work with any projection used in the tiff. However, with severe distortions from WM, the homography may not align the interior pixels well.
        So I still recommend using tiffs with WM projections.

        Anyway the method here is:
            Map query wmTlbr to four points, warp to four tiffPixel corners.
            Get axis-aligned bbox from those four tiffPixel corners.
            Sample that bbox.
            Transform those 4 bbox points back to wmTlbr.
            Find homography that relates the sampled four corners (projected back into wm) to original queried wm four corners.
            Warp with that homography.

        The benchmark at the end of this file suggests this function is 10-20x faster than the GDAL one!

        Implementation is translated from my ancient C++ code.


        WARNING:
        One annoying fact of `RasterIO`/`ReadAsArray` is it fails when asking for any pixel range out of bounds of the raster.
        I try to handle that below by varying `sampledActualTlwh` variable below, along with varying `sw`/`sh` to match the asked-for vs actual GSD.
        This works on one test I tiff I tried it with.

        WARNING:
        While the "zooming out" (minimization) works just as good here as for tiffs (thanks to overviews + bilinear sampling),
        the magnifcation "zooming in" process will look better with the GDAL Warp because it can run an expensive nice looking filter on the pixels,
        while this function must just bilinear interpolate on whatever grid is sampled.

        '''

        sw, sh = int(w * oversample), int(h * oversample)

        wmTlbr = wmTlbr.reshape(-1)
        cornersWm = np.array((
            wmTlbr[0], wmTlbr[1], 0,
            wmTlbr[2], wmTlbr[1], 0,
            wmTlbr[2], wmTlbr[3], 0,
            wmTlbr[0], wmTlbr[3], 0), dtype=np.float64).reshape(4,3)

        if not hasattr(self,'prj_to_wm'):
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(self.dset.GetProjection())
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(3857)
            self.prj_to_wm = osr.CoordinateTransformation(src_srs, dst_srs)
            self.wm_to_prj = osr.CoordinateTransformation(dst_srs, src_srs)

        # print('cornersWm:\n',cornersWm)
        cornersPrj = np.array(self.wm_to_prj.TransformPoints(cornersWm))
        # print('cornersPrj:\n',cornersPrj)
        cornersPrj[:, 2] = 1 # Make homogeneous 2d instead of 3d

        cornersTiff = cornersPrj @ self.pix_from_prj.T
        # print('cornersTiff:\n',cornersTiff)
        tiffTlbr = np.concatenate((cornersTiff.min(0), cornersTiff.max(0)))

        border = cv2.BORDER_REPLICATE

        if 0:
            tiffTlwh = np.array((
                tiffTlbr[0],
                tiffTlbr[1],
                tiffTlbr[2] - tiffTlbr[0],
                tiffTlbr[3] - tiffTlbr[1]))
            # sampledAskedTlwh = np.round(tiffTlwh + (-1,-1, 2,2))
            sampledAskedTlwh = np.round(tiffTlwh + (-.5,-.5, 1,1))

            sampledActualTlwh = sampledAskedTlwh
        else:
            tiffTlbr[:2] -= .5
            tiffTlbr[2:] += .5
            tiffTlbr = np.round(tiffTlbr)
            if tiffTlbr[0] < 0 or tiffTlbr[1] < 0 or tiffTlbr[2] >= self.w or tiffTlbr[3] >= self.h:
                oldw = tiffTlbr[2] - tiffTlbr[0]
                oldh = tiffTlbr[3] - tiffTlbr[1]
                tiffTlbr[0] = max(0, tiffTlbr[0])
                tiffTlbr[1] = max(0, tiffTlbr[1])
                tiffTlbr[2] = min(self.w, tiffTlbr[2])
                tiffTlbr[3] = min(self.h, tiffTlbr[3])
                sw = int(sw * (tiffTlbr[2]-tiffTlbr[0]) / oldw + .5)
                sh = int(sh * (tiffTlbr[3]-tiffTlbr[1]) / oldh + .5)
                border = cv2.BORDER_CONSTANT
            sampledActualTlwh = np.array((
                tiffTlbr[0],
                tiffTlbr[1],
                tiffTlbr[2] - tiffTlbr[0],
                tiffTlbr[3] - tiffTlbr[1]))

        sampledImg = self.readPixelRange(sampledActualTlwh, sw, sh, c)

        if sampledImg is None:
            raise ValueError(f'failed to sample img from pix tlwh {sampledActualTlwh} on dataset of size {self.w} x {self.h}')

        sampledActualTlbr = np.array((
            sampledActualTlwh[0],
            sampledActualTlwh[1],
            sampledActualTlwh[0] + sampledActualTlwh[2],
            sampledActualTlwh[1] + sampledActualTlwh[3]))

        sampledCornersPrj = np.array((
            sampledActualTlbr[0], sampledActualTlbr[1], 1,
            sampledActualTlbr[2], sampledActualTlbr[1], 1,
            sampledActualTlbr[2], sampledActualTlbr[3], 1,
            sampledActualTlbr[0], sampledActualTlbr[3], 1.,), dtype=np.float64).reshape(4,3) @ self.prj_from_pix.T

        sampledCornersWm = np.array(self.prj_to_wm.TransformPoints(sampledCornersPrj))

        # iw, ih = sw - 1, sh - 1
        iw, ih = sw, sh
        inPts = np.array((
            0,0,
            iw,0,
            iw,ih,
            0,ih), dtype=np.float32).reshape(4,2)

        ow, oh = w, h
        outPts = np.array((
            ow * (sampledCornersWm[0,0] - wmTlbr[0]) / (wmTlbr[2] - wmTlbr[0]), oh - (oh * (sampledCornersWm[0,1] - wmTlbr[1]) / (wmTlbr[3]-wmTlbr[1])),
            ow * (sampledCornersWm[1,0] - wmTlbr[0]) / (wmTlbr[2] - wmTlbr[0]), oh - (oh * (sampledCornersWm[1,1] - wmTlbr[1]) / (wmTlbr[3]-wmTlbr[1])),
            ow * (sampledCornersWm[2,0] - wmTlbr[0]) / (wmTlbr[2] - wmTlbr[0]), oh - (oh * (sampledCornersWm[2,1] - wmTlbr[1]) / (wmTlbr[3]-wmTlbr[1])),
            ow * (sampledCornersWm[3,0] - wmTlbr[0]) / (wmTlbr[2] - wmTlbr[0]), oh - (oh * (sampledCornersWm[3,1] - wmTlbr[1]) / (wmTlbr[3]-wmTlbr[1]))), dtype=np.float32).reshape(4,2)

        H = cv2.getPerspectiveTransform(inPts, outPts)

        # WARNING: With BORDER_CONSTANT I get some issues along the right edge. The use of replicate largely resolves -- but this may be a bigger isse on some inputs?
        img = cv2.warpPerspective(sampledImg, H, (w,h), borderMode=border, flags=cv2.INTER_LINEAR)

        return img







if __name__ == '__main__':

    if 1:
        '''
        Test my faster vs GDAL's slower warp functionality.
        Profile the two.
        And show the two images and the abs-diff to verify correctness.
        '''

        raster = GdalRaster('/data/naip/Colorado_SteamBoatSprings_to_Pueblo.tiff')
        # raster = GdalRaster('/data/worldgen/dchd/tiffs/do_n17_8090_31sw.2013.tif') # Not WM, 20x faster!

        aoiTlbr = raster.wmTlbr

        import time

        # My function is 10x faster than GDAL warping the whole thing :)

        pad, wmSz, sampleSz = 0, 1024, 1024
        pad, wmSz, sampleSz = 0, 256, 256

        for y in np.arange(aoiTlbr[1]+pad, aoiTlbr[3]-pad, wmSz):
            for x in np.arange(aoiTlbr[0]+pad, aoiTlbr[2]-pad, wmSz):
                tlbr = np.array((x,y, x+wmSz, y+wmSz))

                # Run multiple times to 'warm up' and be fair about disk cache effects.
                for i in range(3):
                    st = time.time()
                    a = raster.readWmRange(tlbr, sampleSz, sampleSz, 3)
                    atime = time.time() - st

                    st = time.time()
                    b = raster.readWmRangeFaster(tlbr, sampleSz, sampleSz, 3)
                    btime = time.time() - st

                    if i == 2:
                        print(f' a {atime*1000:>5.1f}ms b {btime*1000:>5.2f}ms')

                d = abs(a.mean(axis=-1).astype(np.float32) - b.mean(axis=-1).astype(np.float32)).astype(np.uint8)

                cv2.imshow('a',a)
                cv2.imshow('b',b)
                cv2.imshow('abs diff',d)
                cv2.waitKey(0)
