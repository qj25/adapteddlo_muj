import argparse

def dtd_parse():
    # Create the parser
    parser = argparse.ArgumentParser(description="dlo_testdata for validation tests.")
    # Add arguments with default values
    parser.add_argument('--stiff', type=str, default='adapt', help='Stiffness type: native or adapt [default: adapt]')
    parser.add_argument('--test', type=str, default='lhb', help='Validation test type: lhb or mbi [default: lhb]')
    parser.add_argument('--render', type=str, default=0, help='Render mode: 0 (off) or 1 (on) [default: off]')
    parser.add_argument('--newstart', type=int, default=2, help='Option to re_init pickle of env: 0 or 1 [default: on for lhb, off for mbi]')
    parser.add_argument('--loadresults', type=int, default=0, help='Loads results from latest test: 0 or 1 [default: off]')
    return parser

def tswa_parse():
    # Create the parser
    parser = argparse.ArgumentParser(description="test_shape_w_arm for producing simulation shapes and comparing with real experimental results.")
    # Add arguments with default values
    parser.add_argument('--stiff', type=str, default=None, help='specify stiffness type: native or adapt [default: all]')
    parser.add_argument('--wirecolor', type=str, default=None, help='specify wire color: black, red, or white [default: all]')
    parser.add_argument('--moveid', type=int, default=None, help='specify moveid: 0 to 3')
    parser.add_argument('--render', type=int, default=0, help='Render mode: 0 (off) or 1 (on) [default: off]')
    return parser

def r2spi_parse():
    # Create the parser
    parser = argparse.ArgumentParser(description="real_testdata for doing parameter identification for alpha and beta values of wire.")
    # Add arguments with default values
    parser.add_argument('--stiff', type=str, default='adapt', help='specify stiffness type: native or adapt [default: adapt]')
    parser.add_argument('--wirecolor', type=str, default='white', help='specify wire color: black, red, or white [default: white]')
    parser.add_argument('--testtype', type=str, default='bending', help='which parameter is being identified: bending (alpha) or twisting (beta/alpha)')
    parser.add_argument('--testid', type=str, default='0', help='specify testid (total 5): 0 to 4')
    parser.add_argument('--render', type=int, default=0, help='Render mode: 0 (off) or 1 (on) [default: off]')
    parser.add_argument('--newstart', type=int, default=0, help='Option to re_init pickle of env: 0 or 1 [off]')
    parser.add_argument('--loadresults', type=int, default=0, help='Loads results from latest test: 0 or 1 [default: off]')
    return parser

def svr_parse():
    # Create the parser
    parser = argparse.ArgumentParser(description="simvreal_dlomuj for doing comparing sim and real experiment wire poses.")
    # Add arguments with default values
    parser.add_argument('--wirecolor', type=str, default=None, help='specify wire color to plot: black, red, or white [default: all]')
    parser.add_argument('--moveid', type=int, default=None, help='specify moveid: 0 to 3')
    return parser

def d2p_parse():
    # Create the parser
    parser = argparse.ArgumentParser(description="dlomuj_2Dpos for getting wire 2D pos from image.")
    # Add arguments with default values
    parser.add_argument('--wirecolor', type=str, default='white', help='specify wire color to plot: black, red, or white [default: white]')
    parser.add_argument('--newptselect', type=int, default=0, help='Option to re-select points: 0 or 1 [off]')
    parser.add_argument('--testid', type=str, default='0', help='specify testid (total 5): 0 to 4')
    return parser

def spdt_parse():
    # Create the parser
    parser = argparse.ArgumentParser(description="speed_test for obtaining computation speed.")
    # Add arguments with default values
    parser.add_argument('--newstart', type=int, default=0, help='Option to re_init pickle of env: 0 or 1 [off]')
    return parser