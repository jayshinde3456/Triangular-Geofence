export PATH="/opt/homebrew/bin: $PATH"
export JAVA_HOME=$(/usr/libexec/java_home -v 21.0.1)
export PATH=$JAVA_HOME/bin:$PATH 
export PATH=${PATH}:/usr/local/mysql-8.0.36-macos14-arm64/bin # The following lines have been added by Docker Desktop to enable Docker CLI completions.
fpath=(/Users/jayvijayshinde/.docker/completions $fpath)
autoload -Uz compinit
compinit
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
# End of Docker CLI completions
