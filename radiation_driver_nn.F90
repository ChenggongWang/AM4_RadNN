module radiation_driver_nn_mod

! <OVERVIEW>
!    radiation_driver_nn_mod provides a neural network version of
!    radiaitive transfer module to the radiation driver.
! </OVERVIEW>
use fms_mod,               only: FATAL, NOTE, error_mesg, stdout
use fms2_io_mod,           only: FmsNetcdfFile_t, read_data,&
                                 open_file, close_file
use diag_manager_mod,      only: register_diag_field, send_data, &
                                 diag_manager_init
use time_manager_mod,      only: time_manager_init, time_type, operator(>)
use radiation_driver_types_mod, only: radiation_control_type, &
                                      astronomy_type, &
                                      rad_output_type
use shortwave_driver_mod, only: get_solar_constant
!--------------------------------------------------------------------

implicit none
private

public  radiation_driver_nn_init, &
        test
!        nn_radiation_calc, &
!        produce_rad_diag_nn,&

!----------------------------------------------------------------------
! cgw
! define type for linear layer
! contains weight(2d) and bias(1d)
!----------------------------------------------------------------------
public NN_Linear_layer_type
type :: NN_Linear_layer_type 
    real(kind=4), dimension(:,:), pointer :: weight=>NULL()
    real(kind=4), dimension(:),   pointer :: bias=>NULL()
end type NN_Linear_layer_type
public NN_FC_type
type :: NN_FC_type
    integer :: num_hid_nodes
    integer :: num_layers
    type(NN_Linear_layer_type), dimension(:), pointer:: Layers
end type NN_FC_type

                         contains

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!                     PUBLIC SUBROUTINES
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subroutine radiation_driver_nn_init(rad_nn_para_nc, Rad_NN_FC)
    character(len=32), intent(in) :: rad_nn_para_nc
    type(NN_FC_type),  dimension(:), intent(inout) :: Rad_NN_FC
    character(len=32) :: fmt_str
    type(FmsNetcdfFile_t)       :: Rad_NN_para_fileobj        !< Fms2_io fileobj
    character(len=100), dimension(4) :: rad_nn_para_nc_4filepath
    integer :: nn_size0, nn_size1, inn, outunit
    integer :: nn_num_layers, ilayer
    
    outunit = stdout()
    
    call error_mesg ('radiation_driver_mod',  &
             'Initializing the rad_nn module', NOTE)
    call error_mesg ('radiation_driver_mod',  &
             'Reading NetCDF file to obtain NN parameters. INPUT/'//trim(rad_nn_para_nc), NOTE)
    rad_nn_para_nc_4filepath(1) = "INPUT/"//trim(rad_nn_para_nc)//"lw_csaf_Li5Relu_EY.nc"
    rad_nn_para_nc_4filepath(2) = "INPUT/"//trim(rad_nn_para_nc)//"lw_af_Li5Relu_EY.nc"
    rad_nn_para_nc_4filepath(3) = "INPUT/"//trim(rad_nn_para_nc)//"sw_csaf_Li5Relu_EY.nc"
    rad_nn_para_nc_4filepath(4) = "INPUT/"//trim(rad_nn_para_nc)//"sw_af_Li5Relu_EY.nc"
    
    ! read para file  
    do inn = 1, 4
        if (open_file(Rad_NN_para_fileobj, rad_nn_para_nc_4filepath(inn), "read" )) then
            ! read num of NN layers
            call read_data(Rad_NN_para_fileobj, 'LN', nn_num_layers)

            Rad_NN_FC(inn)%num_layers = nn_num_layers
            allocate(Rad_NN_FC(inn)%Layers(Rad_NN_FC(inn)%num_layers))
            ! read weight and bias for each layer
            do ilayer = 1, nn_num_layers 
                ! read size of each layer
                write (fmt_str, "(A4,I1,I1)") "size", ilayer, 0
                call read_data(Rad_NN_para_fileobj, fmt_str, nn_size0)
                write (fmt_str, "(A4,I1,I1)") "size", ilayer, 1
                call read_data(Rad_NN_para_fileobj, fmt_str, nn_size1)
                allocate(Rad_NN_FC(inn)%Layers(ilayer)%weight(nn_size1,nn_size0)) !fortran reverse order
                allocate(Rad_NN_FC(inn)%Layers(ilayer)%bias(nn_size0))
                write(outunit,*) 'nn_size', nn_size0, nn_size1
                write (fmt_str, "(A1,I1)") "W", ilayer
                call read_data(Rad_NN_para_fileobj, fmt_str, Rad_NN_FC(inn)%Layers(ilayer)%weight(:,:))
                write (fmt_str, "(A1,I1)") "B", ilayer
                call read_data(Rad_NN_para_fileobj, fmt_str, Rad_NN_FC(inn)%Layers(ilayer)%bias(:))
                ! for diagnose purpose
                write(outunit,*) Rad_NN_FC(inn)%Layers(ilayer)%bias 
            end do 
            Rad_NN_FC(inn)%num_hid_nodes = nn_size1
            call close_file(Rad_NN_para_fileobj)
        else
            call error_mesg ('radiation_driver_mod',  &
                 'rad_nn_para_nc file open failed. '//trim(rad_nn_para_nc_4filepath(inn)), FATAL)
        endif
    end do

end subroutine radiation_driver_nn_init

subroutine test ()
    call error_mesg('radiation_driver_nn_mod', 'call test', NOTE)

end subroutine test


!#######################################################################

end module radiation_driver_nn_mod
