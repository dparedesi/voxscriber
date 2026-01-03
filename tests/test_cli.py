import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Add src to path so we can import voxscriber
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from voxscriber import cli

# --- Fixtures ---

@pytest.fixture
def mock_env():
    """Fixture to mock environment variables."""
    with patch.dict(os.environ, {}, clear=True):
        yield

@pytest.fixture
def mock_huggingface_hub():
    """Fixture to mock huggingface_hub module."""
    with patch.dict(sys.modules, {"huggingface_hub": MagicMock()}):
        yield sys.modules["huggingface_hub"]

# --- Tests for _get_ffmpeg_info ---

def test_get_ffmpeg_info_success():
    """Test successful detection of FFmpeg."""
    with patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), \
         patch("subprocess.run") as mock_run:

        mock_run.return_value.stdout = "ffmpeg version 6.1.1 Copyright (c) 2000-2023..."

        path, version = cli._get_ffmpeg_info()
        assert path == "/usr/local/bin/ffmpeg"
        assert version == 6

def test_get_ffmpeg_info_not_found():
    """Test when FFmpeg is not found."""
    with patch("shutil.which", return_value=None):
        path, version = cli._get_ffmpeg_info()
        assert path is None
        assert version is None

def test_get_ffmpeg_info_no_version():
    """Test when FFmpeg is found but version cannot be parsed."""
    with patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), \
         patch("subprocess.run") as mock_run:

        mock_run.return_value.stdout = "unexpected output"

        path, version = cli._get_ffmpeg_info()
        assert path == "/usr/local/bin/ffmpeg"
        assert version is None

def test_get_ffmpeg_info_error():
    """Test when subprocess raises an exception."""
    with patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), \
         patch("subprocess.run", side_effect=Exception("error")):

        path, version = cli._get_ffmpeg_info()
        assert path == "/usr/local/bin/ffmpeg"
        assert version is None

# --- Tests for _check_torchcodec_native_lib ---

def test_check_torchcodec_success():
    """Test successful torchcodec import."""
    with patch.dict(sys.modules, {"torchcodec.decoders": MagicMock()}):
        success, error = cli._check_torchcodec_native_lib()
        assert success is True
        assert error == ""

def test_check_torchcodec_failure():
    """Test failed torchcodec import."""
    # We simulate failure by mocking the module such that accessing AudioDecoder fails
    mock_module = MagicMock()
    # Configure the mock to raise ImportError when AudioDecoder is accessed
    type(mock_module).AudioDecoder = property(fget=lambda self: (_ for _ in ()).throw(ImportError("Mocked import error")))

    with patch.dict(sys.modules, {"torchcodec.decoders": mock_module}):
        success, error = cli._check_torchcodec_native_lib()
        assert success is False
        assert "Mocked import error" in error

# --- Tests for _is_ffmpeg7_keg_only ---

def test_is_ffmpeg7_keg_only():
    with patch("os.path.isdir") as mock_isdir:
        mock_isdir.return_value = True
        assert cli._is_ffmpeg7_keg_only() is True

        mock_isdir.return_value = False
        assert cli._is_ffmpeg7_keg_only() is False

        mock_isdir.assert_called_with(cli.FFMPEG7_LIB_PATH)

# --- Tests for _is_dyld_library_path_set ---

def test_is_dyld_library_path_set(mock_env):
    """Test DYLD_LIBRARY_PATH check."""
    # Case 1: Not set
    assert cli._is_dyld_library_path_set() is False

    # Case 2: Set but missing path
    os.environ["DYLD_LIBRARY_PATH"] = "/some/other/path"
    assert cli._is_dyld_library_path_set() is False

    # Case 3: Set and containing path
    os.environ["DYLD_LIBRARY_PATH"] = f"/foo:{cli.FFMPEG7_LIB_PATH}:/bar"
    assert cli._is_dyld_library_path_set() is True

# --- Tests for Token Management ---

def test_get_hf_token_priority(mock_env, mock_huggingface_hub):
    """Test priority order: CLI > Env > Stored."""
    mock_get_token = mock_huggingface_hub.get_token
    mock_get_token.return_value = "stored_token"

    # 1. CLI token provided
    token = cli._get_hf_token(cli_token="cli_token")
    assert token == "cli_token"

    # 2. Env token provided (no CLI)
    os.environ["HF_TOKEN"] = "env_token"
    token = cli._get_hf_token(cli_token=None)
    assert token == "env_token"

    # 3. Stored token (no CLI, no Env)
    del os.environ["HF_TOKEN"]
    token = cli._get_hf_token(cli_token=None)
    assert token == "stored_token"

    # 4. None found
    mock_get_token.return_value = None
    token = cli._get_hf_token(cli_token=None)
    assert token is None

def test_get_hf_token_source(mock_env, mock_huggingface_hub):
    """Test token source identification."""
    mock_huggingface_hub.get_token.return_value = "stored_token"

    # CLI
    token, source = cli._get_hf_token_source("cli_tok")
    assert token == "cli_tok"
    assert "CLI argument" in source

    # Env
    os.environ["HF_TOKEN"] = "env_tok"
    token, source = cli._get_hf_token_source(None)
    assert token == "env_tok"
    assert "environment variable" in source

    # Stored
    del os.environ["HF_TOKEN"]
    token, source = cli._get_hf_token_source(None)
    assert token == "stored_token"
    assert "huggingface-cli login" in source

    # None
    mock_huggingface_hub.get_token.return_value = None
    token, source = cli._get_hf_token_source(None)
    assert token is None
    assert "not found" in source

def test_validate_hf_token(mock_huggingface_hub):
    """Test token validation."""
    mock_api = mock_huggingface_hub.HfApi.return_value

    # Success
    mock_api.whoami.return_value = {"name": "testuser"}
    is_valid, user = cli._validate_hf_token("valid_token")
    assert is_valid is True
    assert user == "testuser"

    # Invalid token (401)
    mock_api.whoami.side_effect = Exception("401 Unauthorized")
    is_valid, msg = cli._validate_hf_token("invalid_token")
    assert is_valid is False
    assert "invalid or expired" in msg.lower()

    # Other error
    mock_api.whoami.side_effect = Exception("Connection error")
    is_valid, msg = cli._validate_hf_token("error_token")
    assert is_valid is False
    assert "Could not validate" in msg

# --- Tests for check_dependencies ---

@patch("voxscriber.cli._get_ffmpeg_info")
@patch("voxscriber.cli._check_torchcodec_native_lib")
def test_check_dependencies_success(mock_check_tc, mock_get_ffmpeg):
    """Test successful dependency check."""
    mock_get_ffmpeg.return_value = ("/path/ffmpeg", 6)
    mock_check_tc.return_value = (True, "")

    errors = cli.check_dependencies()
    assert len(errors) == 0

@patch("voxscriber.cli._get_ffmpeg_info")
def test_check_dependencies_ffmpeg_missing(mock_get_ffmpeg):
    """Test missing FFmpeg."""
    mock_get_ffmpeg.return_value = (None, None)

    errors = cli.check_dependencies()
    assert len(errors) == 1
    assert "FFmpeg not found" in errors[0]

@patch("voxscriber.cli._get_ffmpeg_info")
def test_check_dependencies_ffmpeg_version_bad(mock_get_ffmpeg):
    """Test bad FFmpeg versions."""
    # Too old
    mock_get_ffmpeg.return_value = ("/path/ffmpeg", 3)
    errors = cli.check_dependencies()
    assert len(errors) == 1
    assert "too old" in errors[0]

    # Too new
    mock_get_ffmpeg.return_value = ("/path/ffmpeg", 8)
    errors = cli.check_dependencies()
    assert len(errors) == 1
    assert "version 4-7 is required" in errors[0]

@patch("voxscriber.cli._get_ffmpeg_info")
@patch("voxscriber.cli._check_torchcodec_native_lib")
@patch("voxscriber.cli._is_ffmpeg7_keg_only")
@patch("voxscriber.cli._is_dyld_library_path_set")
def test_check_dependencies_torchcodec_failure(mock_dyld, mock_keg, mock_check_tc, mock_get_ffmpeg):
    """Test torchcodec failure scenarios."""
    mock_get_ffmpeg.return_value = ("/path/ffmpeg", 6)

    # Scenario 1: Generic error
    mock_check_tc.return_value = (False, "Some random error")
    errors = cli.check_dependencies()
    assert len(errors) == 1
    assert "torchcodec import error" in errors[0]

    # Scenario 2: Library path error (generic)
    mock_check_tc.return_value = (False, "Library not loaded: @rpath/libavutil")
    mock_keg.return_value = False # Not keg only
    errors = cli.check_dependencies()
    assert len(errors) == 1
    assert "torchcodec cannot load FFmpeg" in errors[0]

    # Scenario 3: Keg-only error (needs DYLD_LIBRARY_PATH)
    mock_check_tc.return_value = (False, "Library not loaded: @rpath/libavutil")
    mock_keg.return_value = True
    mock_dyld.return_value = False
    errors = cli.check_dependencies()
    assert len(errors) == 1
    assert "ffmpeg@7 is 'keg-only'" in errors[0]
    assert "DYLD_LIBRARY_PATH" in errors[0]

# --- Tests for Main CLI ---

@patch("voxscriber.cli.check_dependencies")
@patch("voxscriber.cli._get_hf_token")
@patch("voxscriber.cli.DiarizationPipeline")
def test_main_success(mock_pipeline_cls, mock_get_token, mock_check_deps):
    """Test successful main execution."""
    mock_check_deps.return_value = []
    mock_get_token.return_value = "valid_token"

    # Create a dummy file for input
    with patch("pathlib.Path.exists", return_value=True):
        with patch("sys.argv", ["voxscriber", "test.m4a", "--output", "outdir"]):
            cli.main()

    # Verify pipeline config
    mock_pipeline_cls.assert_called_once()
    config = mock_pipeline_cls.call_args[0][0]
    assert config.hf_token == "valid_token"
    assert config.parallel is True # Default

    # Verify processing
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.process.assert_called_once()
    args = mock_pipeline.process.call_args[1]
    assert str(args["output_dir"]) == "outdir"

@patch("voxscriber.cli.check_dependencies")
@patch("voxscriber.cli._get_hf_token")
@patch("voxscriber.cli.DiarizationPipeline")
def test_main_generic_exception(mock_pipeline_cls, mock_get_token, mock_check_deps):
    """Test generic exception handling in main."""
    mock_check_deps.return_value = []
    mock_get_token.return_value = "token"

    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.process.side_effect = Exception("Unexpected error")

    with patch("pathlib.Path.exists", return_value=True):
        with patch("sys.argv", ["voxscriber", "test.m4a"]):
            with pytest.raises(SystemExit) as exc:
                cli.main()
            assert exc.value.code == 1

@patch("voxscriber.cli.check_dependencies")
def test_main_dep_fail(mock_check_deps):
    """Test main fails when dependencies are missing."""
    mock_check_deps.return_value = ["Some error"]

    with patch("sys.argv", ["voxscriber", "test.m4a"]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1

@patch("voxscriber.cli.check_dependencies")
@patch("voxscriber.cli._get_hf_token")
def test_main_no_token(mock_get_token, mock_check_deps):
    """Test main fails when no token is found."""
    mock_check_deps.return_value = []
    mock_get_token.return_value = None

    with patch("pathlib.Path.exists", return_value=True):
        with patch("sys.argv", ["voxscriber", "test.m4a"]):
            with pytest.raises(SystemExit) as exc:
                cli.main()
            assert exc.value.code == 1

# --- Tests for Doctor ---

@patch("voxscriber.cli._get_ffmpeg_info")
@patch("voxscriber.cli._check_torchcodec_native_lib")
@patch("voxscriber.cli._get_hf_token_source")
def test_doctor_all_good(mock_get_token, mock_check_tc, mock_get_ffmpeg):
    """Test doctor passes when everything is good."""
    mock_get_ffmpeg.return_value = ("/bin/ffmpeg", 6)
    mock_check_tc.return_value = (True, "")
    mock_get_token.return_value = ("token", "CLI")

    # Mock validation
    with patch("voxscriber.cli._validate_hf_token", return_value=(True, "user")):
        assert cli.doctor() == 0

@patch("voxscriber.cli._get_ffmpeg_info")
def test_doctor_ffmpeg_missing(mock_get_ffmpeg):
    """Test doctor fails when ffmpeg missing."""
    mock_get_ffmpeg.return_value = (None, None)

    # Mock other checks to pass or be skipped
    with patch("voxscriber.cli._check_torchcodec_native_lib", return_value=(True, "")), \
         patch("voxscriber.cli._get_hf_token_source", return_value=("token", "src")), \
         patch("voxscriber.cli._validate_hf_token", return_value=(True, "user")):

        assert cli.doctor() == 1

@patch("voxscriber.cli._get_ffmpeg_info")
@patch("voxscriber.cli._check_torchcodec_native_lib")
@patch("voxscriber.cli._get_hf_token_source")
@patch("builtins.input", return_value="n") # Don't login
def test_doctor_token_missing(mock_input, mock_get_token, mock_check_tc, mock_get_ffmpeg):
    """Test doctor fails when token missing."""
    mock_get_ffmpeg.return_value = ("/bin/ffmpeg", 6)
    mock_check_tc.return_value = (True, "")
    mock_get_token.return_value = (None, "not found")

    assert cli.doctor() == 1

@patch("voxscriber.cli._get_ffmpeg_info")
@patch("voxscriber.cli._check_torchcodec_native_lib")
@patch("voxscriber.cli._get_hf_token_source")
@patch("voxscriber.cli._is_ffmpeg7_keg_only")
@patch("voxscriber.cli._get_shell_config_file")
@patch("builtins.input", return_value="y") # Say yes to fix
def test_doctor_interactive_fix(mock_input, mock_get_shell, mock_keg, mock_get_token, mock_check_tc, mock_get_ffmpeg):
    """Test doctor interactive fix for keg-only ffmpeg."""
    mock_get_ffmpeg.return_value = ("/bin/ffmpeg", 6)
    # Simulate keg-only failure
    mock_check_tc.return_value = (False, "Library not loaded")
    mock_keg.return_value = True

    # Mock shell config
    mock_shell_path = MagicMock(spec=Path)
    mock_shell_path.name = ".zshrc"
    mock_get_shell.return_value = mock_shell_path

    # Mock other checks
    mock_get_token.return_value = ("token", "src")

    with patch("voxscriber.cli._validate_hf_token", return_value=(True, "user")), \
         patch("voxscriber.cli._check_shell_config_has_dyld_path", return_value=False), \
         patch("builtins.open", mock_open()) as mock_file:

        assert cli.doctor() == 1 # Still returns 1 because check initially failed

        # Verify it tried to write to the file
        mock_file.assert_called_with(mock_shell_path, "a")
        handle = mock_file()
        handle.write.assert_called()
        assert "export DYLD_LIBRARY_PATH" in handle.write.call_args[0][0]

# --- Additional Tests for Helper Functions ---

def test_check_shell_config_has_dyld_path():
    """Test checking shell config for existing export."""
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True

    # Case 1: Found
    mock_path.read_text.return_value = f'export DYLD_LIBRARY_PATH="{cli.FFMPEG7_LIB_PATH}:$DYLD_LIBRARY_PATH"'
    assert cli._check_shell_config_has_dyld_path(mock_path) is True

    # Case 2: Not found
    mock_path.read_text.return_value = 'export PATH="/some/path:$PATH"'
    assert cli._check_shell_config_has_dyld_path(mock_path) is False

    # Case 3: File doesn't exist
    mock_path.exists.return_value = False
    assert cli._check_shell_config_has_dyld_path(mock_path) is False

def test_get_shell_config_file():
    """Test shell config file detection."""
    # Mock home directory
    with patch("pathlib.Path.home", return_value=Path("/home/user")):
        # Zsh
        with patch.dict(os.environ, {"SHELL": "/bin/zsh"}):
            assert cli._get_shell_config_file() == Path("/home/user/.zshrc")

        # Bash - profile exists
        with patch.dict(os.environ, {"SHELL": "/bin/bash"}), \
             patch("pathlib.Path.exists", return_value=True):
            assert cli._get_shell_config_file() == Path("/home/user/.bash_profile")

        # Unknown shell
        with patch.dict(os.environ, {"SHELL": "/bin/fish"}):
            assert cli._get_shell_config_file() is None

# --- Additional Tests for Main Argument Parsing ---

@patch("voxscriber.cli.check_dependencies", return_value=[])
@patch("voxscriber.cli._get_hf_token", return_value="token")
@patch("voxscriber.cli.DiarizationPipeline")
def test_main_arguments_mapping(mock_pipeline_cls, mock_get_token, mock_check_deps):
    """Test comprehensive argument mapping to config."""
    args = [
        "voxscriber", "test.m4a",
        "--model", "tiny",
        "--language", "fr",
        "--speakers", "3",
        "--min-speakers", "2",
        "--max-speakers", "4",
        "--device", "cpu",
        "--sequential",
        "--quiet"
    ]

    with patch("pathlib.Path.exists", return_value=True):
        with patch("sys.argv", args):
            cli.main()

    # Verify config mapping
    config = mock_pipeline_cls.call_args[0][0]
    assert config.whisper_model == "tiny"
    assert config.language == "fr"
    assert config.num_speakers == 3
    assert config.min_speakers == 2
    assert config.max_speakers == 4
    assert config.device == "cpu"
    assert config.parallel is False  # --sequential implies parallel=False
    assert config.verbose is False   # --quiet implies verbose=False

@patch("voxscriber.cli.check_dependencies", return_value=[])
@patch("voxscriber.cli._get_hf_token", return_value="token")
@patch("voxscriber.cli.DiarizationPipeline")
def test_main_print_result(mock_pipeline_cls, mock_get_token, mock_check_deps):
    """Test --print flag."""
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.process.return_value = "The transcript"

    with patch("pathlib.Path.exists", return_value=True):
        with patch("sys.argv", ["voxscriber", "test.m4a", "--print"]):
            cli.main()

    mock_pipeline.print_transcript.assert_called_with("The transcript")

@patch("voxscriber.cli.check_dependencies", return_value=[])
@patch("voxscriber.cli._get_hf_token", return_value="token")
@patch("voxscriber.cli.DiarizationPipeline")
def test_main_keyboard_interrupt(mock_pipeline_cls, mock_get_token, mock_check_deps):
    """Test Ctrl+C handling."""
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.process.side_effect = KeyboardInterrupt()

    with patch("pathlib.Path.exists", return_value=True):
        with patch("sys.argv", ["voxscriber", "test.m4a"]):
            with pytest.raises(SystemExit) as exc:
                cli.main()
            assert exc.value.code == 130

# --- Additional Tests for Doctor Branches ---

@patch("voxscriber.cli._get_ffmpeg_info")
@patch("voxscriber.cli._check_torchcodec_native_lib", return_value=(True, ""))
@patch("voxscriber.cli._get_hf_token_source", return_value=("token", "src"))
@patch("voxscriber.cli._validate_hf_token", return_value=(True, "user"))
def test_doctor_ffmpeg_versions(mock_val, mock_src, mock_tc, mock_ffmpeg):
    """Test doctor with various FFmpeg versions."""

    # Version > 7 (Unsupported)
    mock_ffmpeg.return_value = ("/bin/ffmpeg", 8)
    assert cli.doctor() == 1

    # Version < 4 (Too old)
    mock_ffmpeg.return_value = ("/bin/ffmpeg", 3)
    assert cli.doctor() == 1

    # Version Unknown
    mock_ffmpeg.return_value = ("/bin/ffmpeg", None)
    assert cli.doctor() == 1

@patch("voxscriber.cli._get_ffmpeg_info", return_value=("/bin/ffmpeg", 6))
@patch("voxscriber.cli._check_torchcodec_native_lib", return_value=(True, ""))
@patch("voxscriber.cli._get_hf_token_source")
@patch("voxscriber.cli._run_hf_login")
@patch("voxscriber.cli._get_hf_token")
@patch("voxscriber.cli._validate_hf_token")
@patch("builtins.input", return_value="y")
def test_doctor_login_flow(mock_input, mock_validate, mock_get_token_simple, mock_login, mock_get_token_source, mock_check_tc, mock_get_ffmpeg):
    """Test doctor login flow."""
    # Initial state: No token
    mock_get_token_source.return_value = (None, "not found")

    # Login succeeds
    mock_login.return_value = True

    # After login, token is found
    mock_get_token_simple.return_value = "new_token"
    mock_validate.return_value = (True, "new_user")

    assert cli.doctor() == 0 # Success

    # Verify login was called
    mock_login.assert_called_once()

    # Verify re-check happened
    mock_validate.assert_called_with("new_token")

# --- Tests for _run_hf_login ---

def test_run_hf_login_success(mock_huggingface_hub):
    """Test successful login."""
    assert cli._run_hf_login() is True
    mock_huggingface_hub.login.assert_called_once()

def test_run_hf_login_exception(mock_huggingface_hub):
    """Test login failure."""
    mock_huggingface_hub.login.side_effect = Exception("Login failed")
    assert cli._run_hf_login() is False

def test_main_file_not_found():
    """Test main when audio file does not exist."""
    with patch("pathlib.Path.exists", return_value=False), \
         patch("voxscriber.cli.check_dependencies", return_value=[]):
        with patch("sys.argv", ["voxscriber", "missing.m4a"]):
            with pytest.raises(SystemExit) as exc:
                cli.main()
            assert exc.value.code == 1

@patch("voxscriber.cli._get_ffmpeg_info", return_value=("/bin/ffmpeg", 6))
@patch("voxscriber.cli._check_torchcodec_native_lib", return_value=(True, ""))
@patch("voxscriber.cli._get_hf_token_source", return_value=(None, "not found"))
@patch("builtins.input", side_effect=KeyboardInterrupt)
def test_doctor_login_cancelled(mock_input, mock_get_token, mock_check_tc, mock_get_ffmpeg):
    """Test doctor login cancellation."""
    assert cli.doctor() == 1

@patch("voxscriber.cli._get_ffmpeg_info", return_value=("/bin/ffmpeg", 6))
@patch("voxscriber.cli._check_torchcodec_native_lib", return_value=(False, "Library not loaded"))
@patch("voxscriber.cli._is_ffmpeg7_keg_only", return_value=True)
@patch("voxscriber.cli._get_shell_config_file", return_value=Path(".zshrc"))
@patch("voxscriber.cli._check_shell_config_has_dyld_path", return_value=False)
@patch("voxscriber.cli._get_hf_token_source", return_value=("token", "src"))
@patch("voxscriber.cli._validate_hf_token", return_value=(True, "user"))
@patch("builtins.input", side_effect=EOFError)
def test_doctor_fix_cancelled(mock_input, mock_validate, mock_get_token, mock_check_shell, mock_get_shell, mock_keg, mock_check_tc, mock_get_ffmpeg):
    """Test doctor interactive fix cancellation."""
    assert cli.doctor() == 1



