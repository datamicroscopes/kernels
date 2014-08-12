all: release

.PHONY: release
release:
	@echo "Setting up cmake (release)"
	@python ./cmake/print_cmake_command.py Release
	[ -d release ] || (mkdir release && cd release && eval `python ../cmake/print_cmake_command.py Release`)

.PHONY: relwithdebinfo
relwithdebinfo:
	@echo "Setting up cmake (relwithdebinfo)"
	@python ./cmake/print_cmake_command.py RelWithDebInfo
	[ -d relwithdebinfo ] || (mkdir relwithdebinfo && cd relwithdebinfo && eval `python ../cmake/print_cmake_command.py RelWithDebInfo`)

.PHONY: debug
debug:
	@echo "Setting up cmake (debug)"
	@python ./cmake/print_cmake_command.py Debug
	[ -d debug ] || (mkdir debug && cd debug && eval `python ../cmake/print_cmake_command.py Debug`)

.PHONY: test
test:
	(cd test && nosetests --verbose)

.PHONY: travis_install
travis_install: 
	make relwithdebinfo
	(cd relwithdebinfo && make && make install)
	pip install .

.PHONY: travis_script
travis_script: 
	(cd test && nosetests --verbose -a '!slow')

.PHONY: lint
lint:
	pyflakes microscopes test bin
	pep8 --filename=*.py,*.pyx --ignore=E265 --exclude=bin/vendor/*.py microscopes test bin
