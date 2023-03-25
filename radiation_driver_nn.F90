module radiation_driver_nn_mod

! <OVERVIEW>
!    radiation_driver_nn_mod provides a neural network version of
!    radiaitive transfer module to the radiation driver.
! </OVERVIEW>
use fms_mod,               only: FATAL, NOTE, error_mesg
use fms2_io_mod,           only: open_file
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


                         contains

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!                     PUBLIC SUBROUTINES
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subroutine radiation_driver_nn_init ()

    call error_mesg('radiation_driver_nn_mod', 'call test init', NOTE)

end subroutine radiation_driver_nn_init

subroutine test ()
    call error_mesg('radiation_driver_nn_mod', 'call test', NOTE)

end subroutine test


!#######################################################################

end module radiation_driver_nn_mod
