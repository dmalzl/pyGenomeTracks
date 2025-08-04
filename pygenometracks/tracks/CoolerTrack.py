import scipy.sparse
from matplotlib import colors
import numpy as np
from .CoolerLikeTrack import CoolerLikeTrack
import logging
import itertools

DEFAULT_MATRIX_COLORMAP = 'RdYlBu_r'
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class CoolerTrack(CoolerLikeTrack):
    SUPPORTED_ENDINGS = ['.cool', '.mcool']
    TRACK_TYPE = 'cool_matrix'
    OPTIONS_TXT = CoolerLikeTrack.OPTIONS_TXT + f"""
# depth is the maximum distance that should be plotted.
# If it is more than 125% of the plotted region, it will
# be adjsted to this maximum value.
depth = 100000
file_type = {TRACK_TYPE}
    """
    DEFAULTS_PROPERTIES = dict({'depth': 100000},
                               **CoolerLikeTrack.DEFAULTS_PROPERTIES)
    INTEGER_PROPERTIES = dict({'depth': [1, np.inf]},
                              **CoolerLikeTrack.INTEGER_PROPERTIES)
    
    SPACERBINWIDTH = 0.1
    # The colormap can only be a colormap

    # spacer bins need to be adjusted by binsize
    def init_view_matrix(self, plot_regions):
        nbins = 0
        for chrom, start, end in plot_regions:
            lo, hi = self.clr.extent((chrom, start, end))
            nbins += hi - lo

        nbins += len(plot_regions) - 1
        view_matrix = np.empty((nbins, nbins))
        view_matrix[:] = np.nan

        return view_matrix

    def get_view_matrix(self, plot_regions):
        view_matrix = self.init_view_matrix(plot_regions)
        view_idx = 0
        depth_in_bp = 0
        start_pos_vec = []
        clr_extents = []
        for nspacer, (chrom_region, region_start, region_end) in enumerate(plot_regions):
            continue_plotting, chrom_region = self.check_before_plotting(chrom_region, region_start, region_end)
            if not continue_plotting:
                raise Exception('problems with input data. please check logs')
            
            ext_lo, ext_hi = self.clr.extent((chrom_region, region_start, region_end))
            extent = ext_hi - ext_lo
            if extent < 1:
                self.log.warning("*Warning*\nThere is no data for the region "
                                "considered on the matrix. "
                                "This will generate an empty track!!\n")

                raise Exception(f'Region {chrom_region}:{region_start}-{region_end} too short')
        
            depth_in_bp += extent * self.clr.binsize
            # select only relevant matrix part
            view_start = view_idx
            view_end = view_idx + extent

            matrix_selector = self.clr.matrix(
                balance = self.properties['weight_name'],
                divisive_weights = self.properties['divisive_weights']
            )
            cis_matrix = matrix_selector[ext_lo: ext_hi, ext_lo: ext_hi]
            # add one bin as spacer
            lo = view_start #+ nspacer
            hi = view_end #+ nspacer
            view_matrix[lo: hi, lo: hi] = cis_matrix
            tmp_pos_vec = [i + self.SPACERBINWIDTH * nspacer for i in range(view_start, view_end + 1)]
            start_pos_vec += tmp_pos_vec
            # print(lo, hi, hi-lo, tmp_pos_vec[0], tmp_pos_vec[-1], tmp_pos_vec[-1] - tmp_pos_vec[0], len(tmp_pos_vec))

            if nspacer:
                # iterating backwards to comply with view_idx
                tmp_view_start = view_start
                tmp_nspacer = nspacer
                for trans_ext_lo, trans_ext_hi in clr_extents[::-1]:
                    trans_extent = trans_ext_hi- trans_ext_lo
                    trans_matrix = matrix_selector[trans_ext_lo: trans_ext_hi, ext_lo: ext_hi]
                    # this is relative to the previous iteration
                    # in which view_start = view_end at the end of the iteration
                    trans_lo = tmp_view_start - trans_extent + tmp_nspacer - 1
                    trans_hi = tmp_view_start + tmp_nspacer - 1
                    tmp_nspacer -= 1
                    view_matrix[trans_lo: trans_hi, lo: hi] = trans_matrix

                    tmp_view_start = tmp_view_start - trans_extent

            view_idx = view_end
            clr_extents.append((ext_lo, ext_hi))

        depth = depth_in_bp // self.clr.binsize + (len(plot_regions) - 2) * np.sqrt(self.SPACERBINWIDTH**2 * 2)
        return view_matrix, depth, start_pos_vec

    def plot(self, ax, plot_regions):
        # get rid of this because we actually want the cut for multi region trans contacts
        # expand region to plus depth on both sides
        # to avoid a 45 degree 'cut' on the edges

        # get bin id of start and end of region in given chromosome
        # chr_start_id, chr_end_id = self.hic_ma.getChrBinRange(chrom_region)
        # chr_start = self.hic_ma.cut_intervals[chr_start_id][1]
        # chr_end = self.hic_ma.cut_intervals[chr_end_id - 1][2]
        # start_bp = max(chr_start, region_start - self.properties['depth'])
        # end_bp = min(chr_end, region_end + self.properties['depth'])
        matrix, depth, start_pos_vec = self.get_view_matrix(plot_regions)

        matrix = matrix * self.properties['scale_factor']

        if self.properties['transform'] == 'log1p':
            matrix += 1

        elif self.properties['transform'] in ['-log', 'log']:
            # We first replace 0 values by minimum values after 0
            mask = matrix == 0
            try:
                matrix[mask] = matrix[mask == False].min()
                matrix = np.log(matrix)
            except ValueError:
                self.log.info('All values are 0, no log applied.')
            else:
                if self.properties['transform'] == '-log':
                    matrix = - matrix

        if self.properties['max_value'] is not None:
            vmax = self.properties['max_value']

        else:
            # try to use a 'aesthetically pleasant' max value
            try:
                vmax = np.nanpercentile(matrix.diagonal(1), 80)
            except Exception:
                vmax = None

        if self.properties['min_value'] is not None:
            vmin = self.properties['min_value']
        else:
            # if depth_in_bins > matrix.shape[0]:
            #     # Make sure you keep one bin
            #     depth_in_bins = max(1, matrix.shape[0] - 5)

            # # if the region length is large with respect to the chromosome length, the diagonal may have
            # # very few values or none. Thus, the following lines reduce the number of bins until the
            # # diagonal is at least length 5 but make sure you have at least one value:
            num_bins_from_diagonal = max(1, int(matrix.shape[0]))
            for num_bins in range(0, num_bins_from_diagonal)[::-1]:
                distant_diagonal_values = matrix.diagonal(num_bins)
                if len(distant_diagonal_values) > 5:
                    break

            vmin = np.nanmedian(distant_diagonal_values)

        self.log.info("setting min, max values for track "
                      f"{self.properties['section_name']} to: "
                      f"{vmin}, {vmax}\n")

        if self.properties['transform'] == 'log1p':
            self.current_norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            self.current_norm = colors.Normalize(vmin=vmin, vmax=vmax)

        self.last_img_plotted = self.pcolormesh_45deg(ax, matrix, start_pos_vec)
        if self.properties['rasterize']:
            self.last_img_plotted.set_rasterized(True)
        if self.properties['orientation'] == 'inverted':
            ax.set_ylim(depth, 0)
        else:
            ax.set_ylim(0, depth)

    def pcolormesh_45deg(self, ax, matrix_c, start_pos_vector):
        """
        Turns the matrix 45 degrees and adjusts the
        bins to match the actual start end positions.
        """
        # code for rotating the image 45 degrees
        n = matrix_c.shape[0]
        # create rotation/scaling matrix
        t = np.array([[1, 0.5], [-1, 0.5]])
        # create coordinate matrix and transform it
        matrix_a = np.dot(np.array([(i[1], i[0])
                                    for i in itertools.product(start_pos_vector[::-1],
                                                               start_pos_vector)]), t)
        # this is to convert the indices into bp ranges
        x = matrix_a[:, 1].reshape(n + 1, n + 1)
        y = matrix_a[:, 0].reshape(n + 1, n + 1)
        # plot
        im = ax.pcolormesh(x, y, np.flipud(matrix_c),
                           cmap=self.cmap, norm=self.current_norm)
        return im
